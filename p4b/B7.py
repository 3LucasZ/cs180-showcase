import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from B1 import get_anms_corners
from B2 import extract_feature_descriptor
from B3 import find_matches


def cylindrical_warp(image, focal):
    """Return a cylindrical BGR image, a validity mask, and its center x.

    Inverse mapping: x = f*tan(u/f); y = v/cos(u/f), with all coordinates
    measured from the image center. The output is narrowed to the projected
    horizontal field of view to avoid needless black margins.
    """
    height, width = image.shape[:2]
    cx, cy = (width - 1) / 2, (height - 1) / 2
    half_width = focal * np.arctan(cx / focal)
    out_width = max(1, 2 * int(np.floor(half_width)) + 1)
    out_cx = (out_width - 1) / 2

    u, v = np.meshgrid(
        np.arange(out_width, dtype=np.float32) - out_cx,
        np.arange(height, dtype=np.float32) - cy,
    )
    angle = u / focal
    map_x = focal * np.tan(angle) + cx
    map_y = v / np.cos(angle) + cy

    valid = ((map_x >= 0) & (map_x <= width - 1) &
             (map_y >= 0) & (map_y <= height - 1))
    warped = cv2.remap(image, map_x.astype(np.float32),
                       map_y.astype(np.float32), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    warped[~valid] = 0
    return warped, valid.astype(np.uint8), out_cx


def extract_features(image, cylindrical, valid, focal, out_cx):
    """Reuse Harris/ANMS on the original image and descriptors on the cylinder.

    Detecting corners before warping avoids spurious Harris corners along the
    black/invalid cylindrical boundary. Returns XY points and Nx64 vectors.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, corners = get_anms_corners(gray, verbose=False)  # (2, N), y then x
    height, width = image.shape[:2]
    cx, cy = (width - 1) / 2, (height - 1) / 2

    # Forward-project the source corners into cylindrical coordinates.
    y, x = corners.astype(np.float64)
    u = focal * np.arctan((x - cx) / focal) + out_cx
    v = focal * (y - cy) / np.sqrt((x - cx) ** 2 + focal ** 2) + cy
    points = np.rint(np.column_stack([u, v])).astype(int)  # (x, y)

    # Descriptors use a 40x40 neighborhood: every pixel in it must be valid.
    safe = cv2.erode(valid, np.ones((41, 41), np.uint8))
    gray_cyl = cv2.cvtColor(cylindrical, cv2.COLOR_BGR2GRAY)
    height2, width2 = gray_cyl.shape

    good_pts, descriptors = [], []
    for px, py in points:
        if not (20 <= px < width2 - 20 and 20 <= py < height2 - 20):
            continue
        if not safe[py, px]:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            desc = extract_feature_descriptor(gray_cyl, (py, px)).ravel()
        if np.all(np.isfinite(desc)):
            good_pts.append((px, py))
            descriptors.append(desc)

    if not descriptors:
        return np.empty((0, 2)), np.empty((0, 64))
    return np.array(good_pts, dtype=np.float64), np.array(descriptors)


def translation_from_matches(pts_a, desc_a, pts_b, desc_b,
                             ratio=0.6, tolerance=4.0):
    """Place image B in image A coordinates. One-point translation consensus.

    Each matched pair proposes shift = point_A - point_B. We test every such
    hypothesis (an exhaustive form of one-point RANSAC), then take the median
    of the largest consistent group.
    """
    if len(desc_a) < 1 or len(desc_b) < 2:
        raise ValueError("Too few descriptors to match")
    matches = find_matches(desc_a, desc_b, t=ratio, verbose=False)
    if len(matches) < 4:
        raise ValueError(
            f"Only {len(matches)} feature matches; need at least 4")

    indices = np.array(matches, dtype=int)
    shifts = pts_a[indices[:, 0]] - pts_b[indices[:, 1]]

    # A single correct match determines a proposed translation.
    errors2 = np.sum((shifts[:, None, :] - shifts[None, :, :]) ** 2, axis=2)
    best = np.argmax(np.sum(errors2 < tolerance ** 2, axis=1))
    inliers = errors2[best] < tolerance ** 2
    shift = np.median(shifts[inliers], axis=0)

    # One refinement with respect to the median shift.
    inliers = np.sum((shifts - shift) ** 2, axis=1) < tolerance ** 2
    if np.count_nonzero(inliers) < 4:
        raise ValueError("Translation consensus has fewer than 4 inliers")
    shift = np.median(shifts[inliers], axis=0)
    return shift, len(matches), int(np.count_nonzero(inliers))


def blend_panorama(images, masks, positions, period=None):
    """Feather-blend all cylindrical images in one canvas.

    With period set, copies are placed +/- one period to blend across the
    360-degree seam. No repeated image resampling or cumulative blur.
    """
    positions = np.asarray(positions, dtype=float)
    min_y = int(np.floor(positions[:, 1].min()))
    max_y = int(np.ceil(max(p[1] + im.shape[0]
                            for p, im in zip(positions, images))))
    out_h = max_y - min_y + 1

    if period is None:
        min_x = int(np.floor(positions[:, 0].min()))
        max_x = int(np.ceil(max(p[0] + im.shape[1]
                                for p, im in zip(positions, images))))
        out_w = max_x - min_x + 1
    else:
        out_w = int(round(abs(period)))
        min_x = int(np.floor(positions[:, 0].min()))
        if out_w <= max(im.shape[1] for im in images):
            raise ValueError("Estimated 360-degree period is too short")

    sum_rgb = np.zeros((out_h, out_w, 3), np.float32)
    sum_w = np.zeros((out_h, out_w), np.float32)

    for img, mask, pos in zip(images, masks, positions):
        # Distance-to-invalid pixels provides a simple feathering weight.
        weight = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        weight = np.where(mask > 0, np.clip(
            weight, 1, 40), 0).astype(np.float32)
        x = int(round(pos[0])) - min_x
        y = int(round(pos[1])) - min_y
        h, w = img.shape[:2]

        # Periodic duplicate placements make the seam blendable.
        copies = (x,) if period is None else (x - out_w, x, x + out_w)
        for left in copies:
            src_x0 = max(0, -left)
            src_x1 = min(w, out_w - left)
            if src_x0 >= src_x1:
                continue
            dst_x0 = left + src_x0
            dst_x1 = left + src_x1
            sl = (slice(y, y + h), slice(dst_x0, dst_x1))
            ws = weight[:, src_x0:src_x1]
            sum_rgb[sl] += img[:,
                               src_x0:src_x1].astype(np.float32) * ws[..., None]
            sum_w[sl] += ws

    out = np.zeros_like(sum_rgb, dtype=np.uint8)
    np.divide(sum_rgb, sum_w[..., None], out=sum_rgb,
              where=sum_w[..., None] > 0)
    out[:] = np.clip(sum_rgb, 0, 255).astype(np.uint8)
    return out


def natural_key(path):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", path.name)]


def stitch_folder(folder, scale=0.25, fov=65.0, loop=False,
                  ratio=0.6, tolerance=4.0, pattern=None):
    folder = Path(folder)
    if pattern:
        paths = [p for p in folder.glob(pattern) if p.is_file()]
    else:
        # Prefer 1.JPG, 2.JPG, ... when present: the user's CS180 folder
        # also contains generated outputs like ransac_1.JPG and fd_0.JPG.
        all_photos = [p for p in folder.iterdir() if p.is_file()]
        numbered = [p for p in all_photos if p.stem.isdigit()]
        if len(numbered) >= 2:
            paths = numbered
        else:
            outputs = ("fd_", "ransac_", "matching_", "corners", "auto_",
                       "cylindrical_", "panorama", "mosaic")
            paths = [p for p in all_photos
                     if not p.stem.lower().startswith(outputs)]
    paths = sorted((p for p in paths
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".heic")),
                   key=natural_key)
    if len(paths) < 2:
        raise ValueError("Need at least two overlapping photos in the folder")

    images, masks, points, descriptors = [], [], [], []
    focal = None
    expected_shape = None
    for p in paths:
        original = cv2.imread(str(p))
        if original is None:
            raise ValueError(
                f"Can't read {p.name}; convert HEIC to JPG if needed")
        image = cv2.resize(original, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        if expected_shape is None:
            expected_shape = image.shape
            width = image.shape[1]
            focal = width / (2 * np.tan(np.deg2rad(fov / 2)))
            print(f"Focal length: {focal:.1f} resized-image pixels")
        if image.shape != expected_shape:
            raise ValueError(
                "All photos must have matching dimensions and orientation")

        cylindrical, valid, out_cx = cylindrical_warp(image, focal)
        pts, desc = extract_features(image, cylindrical, valid, focal, out_cx)
        images.append(cylindrical)
        masks.append(valid)
        points.append(pts)
        descriptors.append(desc)
        print(f"{p.name}: {len(desc)} usable features")

    positions = [np.array([0.0, 0.0])]
    for i in range(len(images) - 1):
        shift, nm, ni = translation_from_matches(
            points[i], descriptors[i], points[i + 1], descriptors[i + 1],
            ratio=ratio, tolerance=tolerance)
        positions.append(positions[-1] + shift)
        print(f"{paths[i].name} -> {paths[i+1].name}: "
              f"{ni}/{nm} inliers, shift=({shift[0]:.1f}, {shift[1]:.1f})")

    period = None
    if loop:
        shift, nm, ni = translation_from_matches(
            points[-1], descriptors[-1], points[0], descriptors[0],
            ratio=ratio, tolerance=tolerance)
        closure = positions[-1] + shift
        period = closure[0]
        if abs(period) < images[0].shape[1]:
            raise ValueError(
                "360-degree closure failed: too little horizontal travel")
        print(f"Loop closure: {ni}/{nm} inliers; circumference={abs(period):.1f}px; "
              f"vertical drift={closure[1]:.1f}px")
        # Distribute the accumulated vertical drift over the full revolution.
        positions = [p - np.array([0, closure[1] * i / len(images)])
                     for i, p in enumerate(positions)]

    return blend_panorama(images, masks, positions, period=period)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", default="panorama",
                        help="Folder of consecutively numbered overlapping photos")
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--fov", type=float, default=65.0,
                        help="Approximate horizontal field of view in degrees")
    parser.add_argument("--loop", action="store_true",
                        help="For a complete 360-degree sweep; match last photo to first")
    parser.add_argument("--ratio", type=float, default=0.6,
                        help="B3 squared-distance nearest-neighbor ratio threshold")
    parser.add_argument("--tolerance", type=float, default=4.0,
                        help="Translation inlier tolerance in resized pixels")
    parser.add_argument("--output", default="cylindrical_panorama.jpg")
    parser.add_argument("--pattern", default=None,
                        help='Image filename glob, e.g. "IMG_*.JPG"; overrides auto-detection')
    args = parser.parse_args()

    if args.scale <= 0 or not 1 < args.fov < 170:
        parser.error(
            "scale must be positive and fov between 1 and 170 degrees")
    result = stitch_folder(args.folder, args.scale, args.fov,
                           args.loop, args.ratio, args.tolerance, args.pattern)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), result):
        raise OSError(f"Failed to write {output}")
    print(f"Saved {output} ({result.shape[1]}x{result.shape[0]})")

    # Overview that fits on a typical screen, in addition to the full image.
    factor = min(1.0, 2200 / result.shape[1])
    overview = cv2.resize(result, None, fx=factor, fy=factor,
                          interpolation=cv2.INTER_AREA)
    preview = output.with_name(output.stem + "_preview.jpg")
    cv2.imwrite(str(preview), overview)
    print(f"Saved overview: {preview}")


if __name__ == "__main__":
    main()

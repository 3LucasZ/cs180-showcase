# copying over stuff from A, and re-mastering it if needed.

from scipy.ndimage import distance_transform_edt
import cv2
from pathlib import Path

import numpy as np

# A2 -------------------


def computeH(ref, to_project):
    # find H
    # H = 3x3 matrix with 8DoF since lower right corner is 1
    # such that p' = Hp
    # p = the points we want to project (3 x N) where N = number of correspondences
    # p' = the points we want to project should look like ref after projection (3 x N)
    # expanding all the terms
    # p1 = linear in (h1, ..., h8) with constants (p1, p2, q1, q2) used
    ref = np.astype(ref, np.float64)
    to_project = np.astype(to_project, np.float64)
    A = []
    b = []
    corr = np.hstack((to_project, ref))
    for p1, p2, q1, q2 in corr:
        A.append([p1, p2, 1, 0, 0, 0, -q1*p1, -q1*p2])
        A.append([0, 0, 0, p1, p2, 1, -q2*p1, -q2*p2])
        b.extend([q1, q2])
    x, err, _, _ = np.linalg.lstsq(A, b, rcond=None)
    H = np.append(x, 1)
    # print("Error:", err)
    return H.reshape(3, 3)


# A3 -------------------
def applyH(pts_2d, H):
    pts_3d = np.vstack([pts_2d, np.ones_like(pts_2d[0, :])])
    out_3d = H@pts_3d
    out_2d = out_3d[:2] / out_3d[2:3]
    return out_2d


def t_applyH(pts_2d, H):
    pts_2d = np.transpose(pts_2d, (1, 0))
    pts_3d = np.vstack([pts_2d, np.ones_like(pts_2d[0, :])])
    # print("pts_3d.shape", pts_3d.shape)
    out_3d = H@pts_3d
    out_2d = out_3d[:2] / out_3d[2:3]
    out_2d = np.transpose(out_2d, (1, 0))
    return out_2d


def warpImage(im, H, useNN, bbox=None):
    # Q = [xi yi 1]^T = H * (P=im)
    # P = H^-1 Q
    # output bounding box
    in_height, in_width, _ = im.shape
    in_corners = np.array([[0, 0, in_width-1, in_width-1],
                           [0, in_height-1, in_height-1, 0]])
    if bbox is None:
        warped_corners = applyH(in_corners, H)
        u_min = int(np.floor(warped_corners[0].min()))
        u_max = int(np.ceil(warped_corners[0].max()))
        v_min = int(np.floor(warped_corners[1].min()))
        v_max = int(np.ceil(warped_corners[1].max()))
    else:
        u_min, u_max, v_min, v_max = bbox

    # inverse warp
    u, v = np.meshgrid(
        np.arange(u_min, u_max+1),
        np.arange(v_min, v_max+1)
    )
    out_shape = u.shape
    u = u.ravel()
    v = v.ravel()
    out_pts = np.vstack([u, v])
    Hi = np.linalg.inv(H)
    in_pts = applyH(out_pts, Hi)
    # 2 functions: x(u, v), y(u, v)
    x, y = in_pts[0], in_pts[1]
    # mask of valid input coordinates
    ok = (x >= 0) & (x <= in_width-1) & (y >= 0) & (y <= in_height-1)
    x = x[ok]
    y = y[ok]
    # nearest neighbor interpolation
    out = np.zeros((u.size, 3), dtype=im.dtype)
    if useNN:
        xnn = np.rint(x).astype(int)
        ynn = np.rint(y).astype(int)
        # for (u,v) in out: out[u,v] = im[y(u,v),x(u,v)]
        out[ok] = im[ynn, xnn]
    # bilinear interpolation
    else:
        # filter only valid input coordinates
        x0 = np.floor(x).astype(int)
        x1 = np.minimum(x0+1, in_width-1)
        y0 = np.floor(y).astype(int)
        y1 = np.minimum(y0+1, in_height-1)
        im00 = im[y0, x0]
        im01 = im[y0, x1]
        im10 = im[y1, x0]
        im11 = im[y1, x1]
        # broadcast (N,0) bil weights
        dx = (x-x0)[:, None]
        dy = (y-y0)[:, None]
        out[ok] = (
            (1-dx)*(1-dy)*im00 +
            dx*(1-dy)*im10 +
            (1-dx)*dy*im01 +
            dx*dy*im11
        )
    return out.reshape(out_shape[0], out_shape[1], 3), ok.reshape(out_shape), (u_min, v_min)


def warpImageNearestNeighbor(im, H, bbox=None):
    return warpImage(im, H, True, bbox)


def warpImageBilinear(im, H, bbox=None):
    return warpImage(im, H, False, bbox)

# A4


def lopass(im, sigma):
    if (sigma == 0):
        return im
    ksize = 2 * int(3 * sigma) + 1
    f = cv2.GaussianBlur(
        im,
        (ksize, ksize),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT
    )
    return f


def stack(im, mask=False):
    sigmas = [2, 128] if mask else [0, 4]
    Gstack = []
    for i, sigma in enumerate(sigmas):
        Gstack.append(lopass(im, sigma))
    Lstack = []
    for i in range(len(Gstack)-1):
        Lstack.append(Gstack[i]-Gstack[i+1])
    Lstack.append(Gstack[-1])
    return np.stack(Gstack), np.stack(Lstack)


def colornorm(src, ref):
    out = src.copy()
    for ch in range(3):
        src_u = np.ma.masked_less(src[:, :, ch], 0.001).mean()
        src_s = np.ma.masked_less(src[:, :, ch], 0.001).std()
        ref_u = np.ma.masked_less(ref[:, :, ch], 0.001).mean()
        ref_s = np.ma.masked_less(ref[:, :, ch], 0.001).std()
        out[:, :, ch] = ((src[:, :, ch] - src_u) / (src_s) * ref_s + ref_u)
    return out


def colornorm2(src, ref, maskL, maskR):
    overlap = maskL & maskR
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    diff = ref_gray[overlap].mean() - src_gray[overlap].mean()
    out = src.copy()
    out[maskL] = np.clip(src[maskL] + diff, 0, 1)
    return out


def color_extend(im, mask):
    indices = distance_transform_edt(
        ~mask, return_distances=False, return_indices=True)
    return im[indices[0], indices[1]]


def create_blend_mask(maskL, maskR):
    intersection = maskL & maskR
    y_bbox, x_bbox = np.where(intersection)
    mid = (np.min(x_bbox) + np.max(x_bbox)) // 2
    mask = np.zeros(maskL.shape, dtype=np.float32)
    mask[maskL & ~maskR] = 1
    mask[:, :mid][intersection[:, :mid]] = 1
    return mask


def stitch(frameL, frameR, maskL, maskR):
    mask = create_blend_mask(maskL, maskR)
    # cv2.imwrite(OUT_PATH / "maskL.jpg", mask*255)
    frameL = frameL.astype(np.float32) / 255.0
    frameR = frameR.astype(np.float32) / 255.0
    frameL = colornorm2(frameL, frameR, maskL, maskR)
    # color extension makes the black seams less bad
    _, left_lstack = stack(color_extend(frameL, maskL))
    _, right_lstack = stack(color_extend(frameR, maskR))
    mask_gstack, _ = stack(mask, True)
    mask_gstack = mask_gstack[..., None]  # broadcast (RGB)
    lstack = left_lstack * mask_gstack
    rstack = right_lstack * (1-mask_gstack)
    lrstack = lstack + rstack
    answer = np.sum(lrstack, axis=0)
    # only use the part in the mask, the other stuff might get blended with the black
    LnotR = maskL & ~maskR
    RnotL = maskR & ~maskL
    answer[LnotR] = frameL[LnotR]
    answer[RnotL] = frameR[RnotL]
    # 0 out stuff not in union of masks
    answer[~(maskL | maskR)] = 0
    return np.clip(answer*255, 0, 255).astype(np.uint8)

# A4 remastered


def manual_mosaic2(left, center, left_pts, center_pts):
    # NOTE: this stitches the left image to be in the frame of the center image
    H_left = computeH(center_pts, left_pts)
    warp_left, mask_left, origin_left = warpImageBilinear(left, H_left)
    dataset = [
        (warp_left, mask_left, origin_left),
        (center, np.ones(center.shape[:2], dtype=bool), (0, 0)),
    ]

    # messy coordinates: origin (x, y) image.shape (y, x)
    # bbox for mosaic
    x_min = min(origin[0] for im, mask, origin in dataset)
    y_min = min(origin[1] for im, mask, origin in dataset)
    x_max = max(origin[0] + im.shape[1] for im, mask, origin in dataset)
    y_max = max(origin[1] + im.shape[0] for im, mask, origin in dataset)

    # convert dataset to mosaic coordinate frame
    h_mosaic = y_max - y_min
    w_mosaic = x_max - x_min
    # print("h:", h_mosaic, "w:", w_mosaic)
    frames = []
    masks = []
    for im, mask, (x, y) in dataset:
        h, w = im.shape[:2]
        y_off = y - y_min
        x_off = x - x_min
        frame = np.zeros((h_mosaic, w_mosaic, 3), dtype=im.dtype)
        frame_mask = np.zeros((h_mosaic, w_mosaic), dtype=bool)
        frame[y_off:y_off+h, x_off:x_off+w] = im
        frame_mask[y_off:y_off+h, x_off:x_off+w] = mask
        frames.append(frame)
        masks.append(frame_mask)

    total = stitch(frames[0], frames[1], masks[0], masks[1])
    return total


def manual_mosaic3(left, center, right, left_pts, center_left_pts, center_right_pts, right_pts):
    H_left = computeH(center_left_pts, left_pts)
    H_right = computeH(center_right_pts, right_pts)
    warp_left, mask_left, origin_left = warpImageBilinear(left, H_left)
    warp_right, mask_right, origin_right = warpImageBilinear(right, H_right)
    dataset = [
        (warp_left, mask_left, origin_left),
        (center, np.ones(center.shape[:2], dtype=bool), (0, 0)),
        (warp_right, mask_right, origin_right),
    ]

    # messy coordinates: origin (x, y) image.shape (y, x)
    # bbox for mosaic
    x_min = min(origin[0] for im, mask, origin in dataset)
    y_min = min(origin[1] for im, mask, origin in dataset)
    x_max = max(origin[0] + im.shape[1] for im, mask, origin in dataset)
    y_max = max(origin[1] + im.shape[0] for im, mask, origin in dataset)

    # convert dataset to mosaic coordinate frame
    h_mosaic = y_max - y_min
    w_mosaic = x_max - x_min
    # print("h:", h_mosaic, "w:", w_mosaic)
    frames = []
    masks = []
    for im, mask, (x, y) in dataset:
        h, w = im.shape[:2]
        y_off = y - y_min
        x_off = x - x_min
        frame = np.zeros((h_mosaic, w_mosaic, 3), dtype=im.dtype)
        frame_mask = np.zeros((h_mosaic, w_mosaic), dtype=bool)
        frame[y_off:y_off+h, x_off:x_off+w] = im
        frame_mask[y_off:y_off+h, x_off:x_off+w] = mask

        # f = 0.25
        # frame = cv2.resize(frame, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        # frame_mask = cv2.resize(frame_mask.astype(
        #     np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

        frames.append(frame)
        masks.append(frame_mask)

    # frame order: left, center, right
    part1 = stitch(frames[0], frames[1], masks[0], masks[1])
    total = stitch(part1, frames[2], masks[0] | masks[1], masks[2])
    return total

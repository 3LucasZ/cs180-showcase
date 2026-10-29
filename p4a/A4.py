from scipy.ndimage import distance_transform_edt
import cv2
from pathlib import Path

import numpy as np

from A2 import computeH
import mosaic_1.corr
import mosaic_2.corr
import mosaic_3.corr
from A3 import warpImageBilinear, warpImageNearestNeighbor

DIR = Path(__file__).resolve().parent
t = 1
IN_PATH = DIR / f"mosaic_{t}"
OUT_PATH = DIR / f"mosaic_{t}"
if (t == 1):
    center_left_pts = mosaic_1.corr.center_left_pts
    left_pts = mosaic_1.corr.left_pts
    right_pts = mosaic_1.corr.right_pts
    center_right_pts = mosaic_1.corr.center_right_pts
elif t == 2:
    center_left_pts = mosaic_2.corr.center_left_pts
    left_pts = mosaic_2.corr.left_pts
    right_pts = mosaic_2.corr.right_pts
    center_right_pts = mosaic_2.corr.center_right_pts
elif t == 3:
    center_left_pts = mosaic_3.corr.center_left_pts
    left_pts = mosaic_3.corr.left_pts
    right_pts = mosaic_3.corr.right_pts
    center_right_pts = mosaic_3.corr.center_right_pts

left = cv2.imread(IN_PATH / "left.JPG")
center = cv2.imread(IN_PATH / "center.JPG")
right = cv2.imread(IN_PATH / "right.JPG")

# downscale everything because massive images from iphone are hard to process
f = 0.5
print("left.shape", left.shape)
left = cv2.resize(left, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
center = cv2.resize(center, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
right = cv2.resize(right, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
print("left.shape", left.shape)
center_left_pts = center_left_pts * f
left_pts = left_pts * f
center_right_pts = center_right_pts * f
right_pts = right_pts * f

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
print("h:", h_mosaic, "w:", w_mosaic)
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


# copy from p2/2.3and4/main2.py


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


# def colornorm2(src, ref):
#     src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
#     ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
#     diff = np.ma.masked_less(ref_gray, 0.001).mean(
#     ) - np.ma.masked_less(src_gray, 0.001).mean()
#     return np.clip(src + diff, 0, 1)


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


# frame order: left, center, right
part1 = stitch(frames[0], frames[1], masks[0], masks[1])
total = stitch(part1, frames[2], masks[0] | masks[1], masks[2])
cv2.imwrite(OUT_PATH / "mosaic.blend.jpg", total)

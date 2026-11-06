from pathlib import Path

import cv2
import numpy as np

from B5 import cylindrical_warp
from A import computeH
from B4 import get_corrs

from A import computeH, warpImageBilinear, warpImageNearestNeighbor, stitch
from B4 import get_corrs


def auto_mosaic(ims):
    tot_dx, tot_dy = 0.0, 0.0
    dataset = [(ims[0], (0, 0))]
    for i in range(len(ims)-1):
        im1 = ims[i]
        im2 = ims[i+1]
        pts1, pts2 = get_corrs(im1, im2)
        dx, dy = np.median(pts1 - pts2, axis=0)
        tot_dx, tot_dy = tot_dx + dx, tot_dy + dy
        dataset.append((ims[i+1], (round(tot_dx), round(tot_dy))))
    x_min = min(origin[0] for im, origin in dataset)
    y_min = min(origin[1] for im, origin in dataset)
    x_max = max(origin[0] + im.shape[1] for im, origin in dataset)
    y_max = max(origin[1] + im.shape[0] for im, origin in dataset)

    h_mosaic = y_max - y_min
    w_mosaic = x_max - x_min

    frames = []
    masks = []
    for im, (x, y) in dataset:
        h, w = im.shape[:2]
        y_off = y - y_min
        x_off = x - x_min
        frame = np.zeros((h_mosaic, w_mosaic, 3), dtype=im.dtype)
        frame_mask = np.zeros((h_mosaic, w_mosaic), dtype=bool)
        frame[y_off:y_off+h, x_off:x_off+w] = im
        frame_mask[y_off:y_off+h, x_off:x_off+w] = 1
        frames.append(frame)
        masks.append(frame_mask)

    total = frames[0]
    total_mask = masks[0]
    for i in range(len(frames)-1):
        total = stitch(total, frames[i+1], total_mask, masks[i+1])
        total_mask = total_mask | masks[i+1]

    return total


DIR = Path(__file__).resolve().parent
IN_PATH = DIR / f"panorama"
OUT_PATH = DIR / f"panorama"
N = 10

ims = [cv2.imread(IN_PATH / f"{i+1}.JPG") for i in range(N)]
f = 0.5
ims = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
       for im in ims]
focal = 1514 * f
for i in range(N):
    warp, mask, _ = cylindrical_warp(ims[i], focal)
    ims[i] = warp
    h, w, c = ims[i].shape
    ims[i] = ims[i][h//4:3*h//4, :]


panorama = auto_mosaic(ims)
cv2.imwrite(OUT_PATH / "panorama.jpg", panorama)

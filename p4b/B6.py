from pathlib import Path

import cv2
import numpy as np

from B5 import cylindrical_warp
from A import computeH, manual_mosaic2
from B4 import get_corrs

from A import computeH, warpImageBilinear, warpImageNearestNeighbor, stitch
from B4 import get_corrs


DIR = Path(__file__).resolve().parent
IN_PATH = DIR / f"panorama"
OUT_PATH = DIR / f"panorama"
N = 10

ims = [cv2.imread(IN_PATH / f"{i+1}.JPG") for i in range(N)]
f = 0.25
ims = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
       for im in ims]
focal = 1514 * f
masks = [None] * N
for i in range(N):
    warp, mask, _ = cylindrical_warp(ims[i], focal)
    ims[i] = warp
    masks[i] = mask

# mask = masks[0]
# panorama = ims[0]
# for i, im in enumerate(ims[1:]):
#     print("---STITCH:", i+1)
#     im_mask = masks[i]
#     panorama_mask, panorama = auto_mosaic2(
#         im, im_mask, panorama, panorama_mask)
#     cv2.imwrite(OUT_PATH / f"panorama_{i}.jpg", panorama)
# cv2.imwrite(OUT_PATH / "panorama.jpg", panorama)


# HAD HELP WITH AI HERE:

Hs = [np.eye(3)]

for i in range(1, N):
    pts_prev, pts_curr = get_corrs(ims[i-1], ims[i])

    # Map current image -> previous image
    # H = computeH(pts_prev, pts_curr)
    # ABOVE H IS NOT WORKING, SO TRY TRANSLATION ONLY!!!
    dx, dy = np.median(pts_prev - pts_curr, axis=0)
    H = np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])

    # Compose with previous transformation
    Hs.append(Hs[-1] @ H)


data = []

for im, mask, H in zip(ims, masks, Hs):
    warped, inside, (x, y) = warpImageBilinear(im, H)
    h, w = warped.shape[:2]

    # Transform the cylindrical validity mask
    mask_rgb = cv2.cvtColor(
        mask.astype(np.uint8) * 255,
        cv2.COLOR_GRAY2BGR
    )
    warped_mask, _, _ = warpImageNearestNeighbor(
        mask_rgb, H, bbox=(x, x+w-1, y, y+h-1)
    )

    valid = inside & (warped_mask[:, :, 0] > 0)
    data.append((warped, valid, x, y))

# Global canvas dimensions
xmin = min(x for im, mask, x, y in data)
ymin = min(y for im, mask, x, y in data)
xmax = max(x + im.shape[1] for im, mask, x, y in data)
ymax = max(y + im.shape[0] for im, mask, x, y in data)

W = xmax - xmin
H = ymax - ymin

# Stitchhh
panorama = np.zeros((H, W, 3), dtype=np.uint8)
panorama_mask = np.zeros((H, W), dtype=bool)

for im, mask, x, y in data:
    h, w = im.shape[:2]
    x -= xmin
    y -= ymin

    frame = np.zeros_like(panorama)
    frame_mask = np.zeros_like(panorama_mask)

    frame[y:y+h, x:x+w] = im
    frame_mask[y:y+h, x:x+w] = mask

    if not panorama_mask.any():
        panorama = frame
    else:
        panorama = stitch(
            panorama, frame,
            panorama_mask, frame_mask
        )

    panorama_mask |= frame_mask

cv2.imwrite(OUT_PATH / "panorama.jpg", panorama)

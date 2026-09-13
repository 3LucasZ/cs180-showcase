import numpy as np
import cv2
from pathlib import Path

DIR = Path(__file__).resolve().parent
IN_DIR = DIR / "in"
OUT_DIR = DIR / "out"


def downsample2(im):
    # crop image to be even size before downsampling by 2
    if (im.shape[0] % 2 == 1):
        im = im[1:, :]
    if (im.shape[1] % 2 == 1):
        im = im[:, 1:]
    im = im.astype(np.float16)
    return (im[0::2, 0::2] + im[0::2, 1::2] + im[1::2, 0::2] + im[1::2, 1::2]) / 4.0


def downsample2gauss(im):
    im = cv2.pyrDown(im)
    return im


IN_PATH = IN_DIR / f"sample{1}" / "cathedral.jpg"
OUT_PATH = OUT_DIR / f"test" / "cathedral_ds_cmp.jpg"
im = cv2.imread(IN_PATH, cv2.IMREAD_GRAYSCALE)
height = np.floor(im.shape[0] / 3.0).astype(np.uint)
g = im[height: 2*height]
l = downsample2(g)
r = downsample2gauss(g)
h = min(l.shape[0], r.shape[0])
w = min(l.shape[1], r.shape[1])
l = l[:h, :w]
r = r[:h, :w]
im_out = np.hstack([l, r])
cv2.imwrite(OUT_PATH, im_out)

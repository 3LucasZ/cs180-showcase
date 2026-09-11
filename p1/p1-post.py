# NOTE: IM A CS180 STUDENT NOT CS280 SO BELLS + WHISTLES ARE OPTIONAL.
# got external help for this section

from pathlib import Path

import cv2
import numpy as np


def autocrop_frame(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gx = np.abs(cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3))
    h, w = im.shape
    vs = gx.mean(axis=0)
    hs = gy.mean(axis=1)
    sx = int(w * 0.2)
    sy = int(h * 0.2)
    l = np.argmax(vs[:sx])
    r = w - sx + np.argmax(vs[w-sx:])
    t = np.argmax(hs[:sy])
    b = h - sy + np.argmax(hs[h-sy:])
    p = 10
    l += w//p
    r -= w//p
    t += h//p
    b -= h//p
    return t, b, l, r


def expand_chs(im):
    chs = cv2.split(im)
    new_chs = []
    for ch in chs:
        new_chs.append(expand_ch(ch))
    return cv2.merge(new_chs)


def expand_ch(ch):
    # a = np.percentile(im, 1)
    # b = np.percentile(im, 99)
    # im = (im - a) / (b - a)
    # return np.clip(im, 0, 1)
    ch = cv2.normalize(ch, None, alpha=0, beta=1,
                       norm_type=cv2.NORM_MINMAX)
    return ch


DIR = Path(__file__).resolve().parent
IN_PATH = DIR / "out" / "sample3" / "lake.jpg"
OUT_PATH = DIR / "out" / "test" / "expand_ch.jpg"
im = cv2.imread(IN_PATH)
print("input image:", IN_PATH, im.shape, im.dtype)
im = im.astype(np.float32)/255
im = expand_chs(im)
cv2.imwrite(OUT_PATH, im)

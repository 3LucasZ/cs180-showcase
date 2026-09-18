from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from align_image_code import align_images
from scipy.signal import convolve2d

DIR = Path(__file__).resolve().parent
IN_PATH = DIR / "in"
OUT_PATH = DIR / "out"


def lopass(im, sigma):
    g = cv2.getGaussianKernel(ksize=5, sigma=sigma, ktype=cv2.CV_32F)
    G = np.outer(g, g)
    f = np.zeros_like(im, dtype=np.float16)
    for ch in range(3):
        f[:, :, ch] = convolve2d(im[:, :, ch], G, mode="same",
                                 boundary="fill", fillvalue=0)
    return f


def hipass(im, sigma):
    f = lopass(im, sigma)
    f = im - f
    return f


def hybrid_image(im1, im2, sigma1, sigma2):
    lo = lopass(im1, sigma1)
    hi = hipass(im2, sigma2)
    im = (lo + hi) / 2
    return im


def test(f1, f2):
    # high frequency, dominates from close up
    im1 = cv2.imread(IN_PATH / f1) / 255.
    # low frequency, dominates from far away
    im2 = cv2.imread(IN_PATH / f2) / 255.
    # Next align images (this code is provided, but may be improved)
    im1_aligned, im2_aligned = align_images(im1, im2)
    # sigma1, sigma2 are cutoff values for the high and low frequencies
    sigma1 = 1
    sigma2 = 1
    hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)
    cv2.imwrite(OUT_PATH / f'{f1}x{f2}.jpg', hybrid)
    cv2.imsave(hybrid)

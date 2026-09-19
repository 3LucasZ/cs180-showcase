from pathlib import Path

import cv2
import numpy as np
from align_image_code import align_images
from scipy.signal import convolve2d

DIR = Path(__file__).resolve().parent
IN_PATH = DIR / "in"
OUT_PATH = DIR / "out"


def lopass(im, sigma):
    ksize = 2 * int(3 * sigma) + 1
    g = cv2.getGaussianKernel(ksize=ksize, sigma=sigma, ktype=cv2.CV_32F)
    G = np.outer(g, g)
    f = np.zeros_like(im, dtype=np.float32)
    for ch in range(3):
        f[:, :, ch] = convolve2d(im[:, :, ch], G, mode="same", boundary="symm")
    return f


def hipass(im, sigma):
    f = lopass(im, sigma)
    f = im - f
    return f


# im1 = smooth image, im2 = detailed image
def hybrid_image(im1, im2, sigma1, sigma2):
    lo = lopass(im1, sigma1)
    hi = hipass(im2, sigma2)
    # im = (lo + hi) / 2 # averaging gives bad results
    im = lo + hi
    im = np.clip(im, 0, 1)
    return im


def test1(f1, f2, out):
    # lopass, low frequency, dominates from far away
    im1 = cv2.imread(IN_PATH / f1).astype(np.float32) / 255.
    im1 = cv2.resize(im1, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    # high pass, high frequency, dominates from close up
    im2 = cv2.imread(IN_PATH / f2).astype(np.float32) / 255.
    im2 = cv2.resize(im2, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    # Next align images (this code is provided, but may be improved)
    im1_aligned, im2_aligned = align_images(im1, im2)
    # sigma1, sigma2 are cutoff values for the high and low frequencies
    sigma1 = 5  # larger -> more blur / less high frequencies
    sigma2 = 5  # larger -> broader range of frequencies in high-pass
    hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)
    cv2.imwrite(
        OUT_PATH / out, hybrid * 255)


# test1("panda.jpg", "redpanda.jpg", "panda.jpg")
# test1("alaska.jpg", "skyline.jpg", "alaska.jpg")
# test1("skyline.jpg", "alaska.jpg", "alaska.jpg")
# test1("waterfall.jpg", "taipei.jpg", "waterfall.jpg")
# test1("taipei.jpg", "waterfall.jpg", "taipei.jpg")
# test1("laplace.jpg", "fourier.jpg", "fourier.jpg")
test1("tdr.jpg", "lincoln.jpg", "lincoln.jpg")


def test2(f):
    im = cv2.imread(IN_PATH / f).astype(np.float32) / 255.
    H, W, C = im.shape
    # lopass, low frequency, dominates from far away
    im1 = im[:H//2, :]
    # high pass, high frequency, dominates from close up
    im2 = im[H//2:, :]
    if (im2.shape[0] > im1.shape[0]):
        im2 = im2[1:, :]
    # Next align images (this code is provided, but may be improved)
    # im1_aligned, im2_aligned = align_images(im1, im2)
    # sigma1, sigma2 are cutoff values for the high and low frequencies
    sigma1 = 2  # larger -> more blur / less high frequencies
    sigma2 = 5  # larger -> more edges / more high frequencies

    im1, im2 = im2, im1
    hybrid = hybrid_image(im1, im2, sigma1, sigma2)
    cv2.imwrite(OUT_PATH / f, hybrid * 255)


# test2("mannahatta.jpg")
# test2("fire.jpg")


def test3():
    # f = "forestcity.png"
    f = "env.jpg"
    im = cv2.imread(IN_PATH / f).astype(np.float32) / 255.
    H, W, C = im.shape
    # lopass, low frequency, dominates from far away
    im1 = im[:, :W//2]
    # high pass, high frequency, dominates from close up
    im2 = im[:, W//2:]
    if (im2.shape[1] > im1.shape[1]):
        im2 = im2[:, 1:]
    # im1 = im1[H//4:, :]
    # im2 = im2[H//4:, :]
    # Next align images (this code is provided, but may be improved)
    # im1_aligned, im2_aligned = align_images(im1, im2)
    # sigma1, sigma2 are cutoff values for the high and low frequencies
    sigma1 = 5  # larger -> more blur / less high frequencies
    sigma2 = 5  # larger -> more edges / more high frequencies
    im1, im2 = im2, im1
    hybrid = hybrid_image(im1, im2, sigma1, sigma2)
    cv2.imwrite(OUT_PATH / f, hybrid * 255)


# test3()

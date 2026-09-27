# Convolutions from Scratch
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from pathlib import Path
import cv2
import numpy as np


DIR = Path(__file__).resolve().parent
IN_PATH = DIR / "in" / "taj.jpg"
OUT_PATH = DIR / "out"
im = cv2.imread(IN_PATH)
im = im.astype(np.float32) / 255.0
print("input image:", IN_PATH, im.shape, im.dtype)


def gen_g(d, sg):
    return cv2.getGaussianKernel(ksize=d, sigma=sg, ktype=cv2.CV_32F)


def gen_G(d, sg):
    g = gen_g(d, sg)
    return np.outer(g, g)


def blur(im, sigma):
    ksize = 2 * int(3 * sigma) + 1
    # G = gen_G(ksize, sigma)
    # out = np.zeros_like(im, dtype=np.float32)
    # for ch in range(3):
    #     out[:, :, ch] = convolve2d(im[:, :, ch], G, mode="same",
    #                                boundary="fill", fillvalue=0)
    f = cv2.GaussianBlur(
        im,
        (ksize, ksize),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT
    )
    return f


def sharpen(im, sigma, a):
    ksize = 2 * int(3 * sigma) + 1
    G = gen_G(ksize, sigma)
    # identity kernel, e
    e = np.zeros_like(G)
    e[G.shape[0] // 2, G.shape[1] // 2] = 1
    # a = 1.0
    unsharp_kern = (1 + a) * e - a * G
    # lesson learned: must apply to all 3 channels...
    # FAIL: make it a 3d kernel (mathematically doesn't work lol)
    # unsharp_kern_3d = np.broadcast_to(
    #     unsharp_kern, (unsharp_kern.shape[0], unsharp_kern.shape[1], 3))
    # im = convolve2d(im, unsharp_kern_3d, mode="same",
    #                 boundary="fill", fillvalue=0)
    out = np.zeros_like(im, dtype=np.float32)
    for ch in range(3):
        out[:, :, ch] = convolve2d(im[:, :, ch], unsharp_kern, mode="same",
                                   boundary="fill", fillvalue=0)
    return out


# for sigma in [1, 3, 10]:
#     for a in [1, 3, 10]:
#         out = sharpen(im, sigma=sigma, a=a)
#         cv2.imwrite(OUT_PATH / f"taj_s{sigma}_a{a}.jpg", out*255)


# im = cv2.imread(DIR / "in" / "falls.jpg")
# im = im.astype(np.float32) / 255.0
# out = sharpen(im, sigma=2, a=4)
# cv2.imwrite(DIR / "out" / "falls.jpg", out*255)

# im = cv2.imread(DIR / "in" / "fox.jpg")
# im = im.astype(np.float32) / 255.0
# out = sharpen(im, sigma=2, a=8)
# cv2.imwrite(DIR / "out" / "fox.jpg", out*255)

im = cv2.imread(DIR / "in" / "joshua.jpg")
# blur
im = im.astype(np.float32) / 255.0
out = blur(im, sigma=5)
cv2.imwrite(DIR / "out" / "joshua.blur.jpg", out*255)
# sharp
out = sharpen(out, sigma=5, a=3)
cv2.imwrite(DIR / "out" / "joshua.sharp.jpg", out*255)

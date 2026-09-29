# Convolutions from Scratch
from scipy.signal import convolve2d
from pathlib import Path
import cv2
import numpy as np
import time


def conv4(im, kern):
    # heights, widths
    H, W = im.shape
    KH, KW = kern.shape
    # paddings; "same" pad
    PT = (KH-1)//2
    PB = KH-1-PT
    PL = (KW-1)//2
    PR = KW-1-PL
    pim = np.pad(im, ((PT, PB), (PL, PR)), mode="constant", constant_values=0)
    out = np.zeros_like(im, dtype=float)
    # conv flips cross-correlation
    kern = np.flip(np.flip(kern, axis=0), axis=1)
    for y in range(H):
        for x in range(W):
            for ky in range(KH):
                for kx in range(KW):
                    out[y, x] += pim[y+ky, x+kx] * kern[ky, kx]
    return out


def conv2(im, kern):
    # heights, widths
    H, W = im.shape
    KH, KW = kern.shape
    # paddings; "same" pad
    PT = (KH-1)//2
    PB = KH-1-PT
    PL = (KW-1)//2
    PR = KW-1-PL
    pim = np.pad(im, ((PT, PB), (PL, PR)), mode="constant", constant_values=0)
    out = np.zeros_like(im, dtype=float)
    # conv flips cross-correlation
    kern = np.flip(np.flip(kern, axis=0), axis=1)
    for y in range(H):
        for x in range(W):
            # parallelize dot product with 0 loops instead of 2
            out[y, x] = np.sum(pim[y:y+KH, x:x+KW] * kern)
    return out


test_im = np.random.random((50, 50))*100
test_kern = np.random.random((9, 9))*100
t0 = time.time()
test_2 = conv2(test_im, test_kern)
t1 = time.time()
print(f"conv2 runtime:{t1-t0}")
test_4 = conv4(test_im, test_kern)
t2 = time.time()
print(f"conv4 runtime:{t2-t1}")
test_sol = convolve2d(test_im, test_kern, mode="same",
                      boundary="fill", fillvalue=0)
t3 = time.time()
print(f"scipy runtime:{t3-t2}")


print("Error on conv2:", np.sum(np.abs(test_sol - test_2)))
# print("Runtime on conv2:")
print("Error on conv4:", np.sum(np.abs(test_sol - test_4)))
# print("Runtime on conv4:")

# DIR = Path(__file__).resolve().parent
# IN_PATH = DIR / "in" / "selfie.png"
# OUT_PATH = DIR / "out"
# im = cv2.imread(IN_PATH, cv2.IMREAD_GRAYSCALE)
# im = cv2.resize(im, (0, 0), fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
# cv2.imwrite(OUT_PATH / "selfie.png", im)

# im = im.astype(np.float16) / 255.0
# print("input image:", IN_PATH, im.shape, im.dtype)


# def post(im):
#     # post process
#     im = np.abs(im)
#     im = im / np.max(im) * 255
#     im = im.astype(np.uint8)
#     return im


# box_kern = np.ones((9, 9)) / 81
# out_box = post(conv2(im, box_kern))
# cv2.imwrite(OUT_PATH / "selfie.box.png", out_box)

# dx_kern = np.array([[1, 0, -1]])
# dy_kern = np.array([[1], [0], [-1]])
# out_dx = post(conv2(im, dx_kern))
# out_dy = post(conv2(im, dy_kern))
# cv2.imwrite(OUT_PATH / "selfie.dx.png", out_dx)
# cv2.imwrite(OUT_PATH / "selfie.dy.png", out_dy)

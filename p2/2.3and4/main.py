from pathlib import Path
import cv2
import numpy as np
from scipy.signal import convolve2d

DIR = Path(__file__).resolve().parent
IN_PATH = DIR / "in"
OUT_PATH = DIR / "out"


def lopass(im, sigma):
    if (sigma == 0):
        return im
    ksize = 2 * int(3 * sigma) + 1

    # TOO SLOW!!!
    # g = cv2.getGaussianKernel(ksize=ksize, sigma=sigma, ktype=cv2.CV_32F)
    # G = np.outer(g, g)
    # f = np.zeros_like(im, dtype=np.float32)
    # for ch in range(im.shape[2]):
    #     f[:, :, ch] = convolve2d(
    #         im[:, :, ch], G, mode="same", boundary="symm")
    # return f
    # BLAZINGLY FAST!!!
    f = cv2.GaussianBlur(
        im,
        (ksize, ksize),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT
    )
    return f


def stack(im):
    sigmas = [0, 2, 4, 8, 16, 32]
    Gstack = []
    for i, sigma in enumerate(sigmas):
        g = lopass(im, sigma)
        # print("i", "g.shape", i, g.shape)
        Gstack.append(g)
    Lstack = []
    for i in range(len(Gstack)-1):
        Lstack.append(Gstack[i]-Gstack[i+1])
    Lstack.append(Gstack[-1])
    # print(Gstack)
    return np.stack(Gstack), np.stack(Lstack)


# trying to make laplacian image viewable and not just black
def norm2(im):
    # return (im-im.min())/im.max()
    # return np.clip(im+0.5, 0, 1)
    # return np.clip(im, 0, 1)
    mx = np.max(np.abs(im))
    return 0.5 + 0.5 * im / mx


def main():
    apple = cv2.imread(IN_PATH / "apple.jpg").astype(np.float32) / 255.0
    orange = cv2.imread(IN_PATH / "orange.jpg").astype(np.float32) / 255.0
    H, W, C = apple.shape
    mask = np.zeros((H, W), dtype=np.float32)
    mask[:, :W//2] = 1.0
    print("apple.shape", apple.shape)
    print("mask.shape", mask.shape)
    left_gstack, left_lstack = stack(apple)
    right_gstack, right_lstack = stack(orange)
    mask_gstack, mask_lstack = stack(mask)
    mask_gstack = mask_gstack[..., None]
    lstack = left_lstack * mask_gstack
    rstack = right_lstack * (1-mask_gstack)
    lrstack = lstack + rstack

    for i, im in enumerate(mask_gstack):
        cv2.imwrite(OUT_PATH / f"mask{i}.jpg", im*255)

    for i, im in enumerate(lstack):
        # normalize
        im = norm2(im)
        cv2.imwrite(OUT_PATH / f"lstack{i}.jpg", im*255)

    cv2.imwrite(OUT_PATH / f"lstack.jpg", np.sum(lstack*255, axis=0))

    for i, im in enumerate(rstack):
        # normalize
        im = norm2(im)
        cv2.imwrite(OUT_PATH / f"rstack{i}.jpg", im*255)

    cv2.imwrite(OUT_PATH / f"rstack.jpg", np.sum(rstack*255, axis=0))

    for i, im in enumerate(lrstack):
        # normalize
        im = norm2(im)
        cv2.imwrite(OUT_PATH / f"lrstack{i}.jpg", im*255)

    cv2.imwrite(OUT_PATH / f"lrstack.jpg", np.sum(lrstack*255, axis=0))


if __name__ == "__main__":
    main()

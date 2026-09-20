from pathlib import Path
import cv2
import numpy as np
from scipy.signal import convolve2d

DIR = Path(__file__).resolve().parent
IN_PATH = DIR / "in2"
OUT_PATH = DIR / "out2"


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
    sigmas = np.array([0, 2, 4, 8, 16, 32, 64])
    if (mask):
        sigmas = sigmas[1:]
        sigmas = np.append(sigmas, 128)
        sigmas *= 2
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
        src_u = src[:, :, ch].mean()
        src_s = src[:, :, ch].std()
        ref_u = ref[:, :, ch].mean()
        ref_s = ref[:, :, ch].std()
        out[:, :, ch] = ((src[:, :, ch] - src_u) / (src_s) * ref_s + ref_u)
    return out


def main(name1, name2, name3):
    l = cv2.imread(IN_PATH / name1).astype(np.float32) / 255.0
    H, W, C = l.shape
    r = cv2.imread(IN_PATH / name2).astype(np.float32) / 255.0
    r = cv2.resize(r, (W, H))
    # r = colornorm(r, l)
    l = colornorm(l, r)

    mask = np.zeros((H, W), dtype=np.float32)
    mask[:, :W//2] = 1.0
    _, left_lstack = stack(l)
    _, right_lstack = stack(r)

    mask_gstack, _ = stack(mask, True)
    mask_gstack = mask_gstack[..., None]
    lstack = left_lstack * mask_gstack
    rstack = right_lstack * (1-mask_gstack)
    lrstack = lstack + rstack
    answer = np.sum(lrstack, axis=0)
    answer = np.clip(answer, 0, 1)
    cv2.imwrite(OUT_PATH / name3, (answer*255).astype(np.uint8))


# main("orange.png", "apple.png", "oraple.png")

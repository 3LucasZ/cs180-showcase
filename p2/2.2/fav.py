from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import numpy as np
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


def fft(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    F = np.fft.fft2(im)
    F = np.fft.fftshift(F)
    mags = np.log(np.abs(F))
    return mags


def main():
    f = "forestcity.png"
    im = cv2.imread(IN_PATH / f).astype(np.float32) / 255.
    H, W, C = im.shape
    # lopass, low frequency, dominates from far away: city
    im1 = im[:, W//2:]
    # hipass, high frequency, dominates from close up: forest
    im2 = im[:, :W//2]
    if (im1.shape[1] > im2.shape[1]):
        im1 = im1[:, 1:]
    sigma1 = 6
    sigma2 = 6
    lo = lopass(im1, sigma1)
    hi = hipass(im2, sigma2)
    hybrid = lo + hi
    cv2.imwrite(OUT_PATH / f, hybrid * 255)

    ffts = [fft(im1), fft(im2), fft(lo), fft(hi), fft(hybrid)]

    # Graphing
    # mn = min(f.min() for f in ffts)
    # mx = max(f.max() for f in ffts)
    all = np.concatenate([f.ravel() for f in ffts])
    mn = np.percentile(all, 1)
    mx = np.percentile(all, 99)
    ts = ["Image_1", "Image_2", "Image_1_lowpass",
          "Image_2_highpass", "Hybrid"]
    for i, (f, t) in enumerate(zip(ffts, ts)):
        plt.figure()
        plt.imshow(f, cmap="inferno", vmin=mn, vmax=mx)
        plt.title(t)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(t+".jpg")


main()

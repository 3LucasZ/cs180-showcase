# Convolutions from Scratch
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from pathlib import Path
import cv2
import numpy as np


DIR = Path(__file__).resolve().parent
IN_PATH = DIR / "in" / "taj.png"
OUT_PATH = DIR / "out"
im = cv2.imread(IN_PATH)
im = im.astype(np.float16) / 255.0
print("input image:", IN_PATH, im.shape, im.dtype)


def gaussian_kernel(d, var):
    k = np.arange(-(d // 2), d // 2 + 1)
    x, y = np.meshgrid(ax, ax)

    g = np.exp(-(x**2 + y**2) / (2 * var**2))

    return g / g.sum()


g_kern =
unsharp_kern =
dx_kern = np.array([[1, 0, -1]])
dy_kern = np.array([[1], [0], [-1]])


out_dx = convolve2d(im, dx_kern, mode="same",
                    boundary="fill", fillvalue=0)
cv2.imwrite(OUT_PATH / "cameraman.dx.png", np.abs(out_dx) * 255)


out_dy = convolve2d(im, dy_kern, mode="same",
                    boundary="fill", fillvalue=0)
cv2.imwrite(OUT_PATH / "cameraman.dy.png", np.abs(out_dy) * 255)


out_g = np.sqrt(out_dy**2 + out_dx**2)
cv2.imwrite(OUT_PATH / "cameraman.g.png", out_g * 255)


# elbow plot
plt.hist(out_g.flatten(), bins=30, cumulative=True,
         density=True, linewidth=4, histtype='step')
plt.title('Elbow Plot')
plt.xlabel('Gradients')
plt.ylabel('Cumulative Probability')
plt.savefig(OUT_PATH / "cameraman.elbow.png")
thresh = 0.3
out_gt = (out_g > thresh) * np.ones(out_g.shape)
cv2.imwrite(OUT_PATH / "cameraman.gt.png", out_gt * 255)

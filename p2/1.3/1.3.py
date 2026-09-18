from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from pathlib import Path
import cv2
import numpy as np


DIR = Path(__file__).resolve().parent
IN_PATH = DIR / "in" / "cameraman.png"
OUT_PATH = DIR / "out"
im = cv2.imread(IN_PATH, cv2.IMREAD_GRAYSCALE)
im = im.astype(np.float16) / 255.0
dx_kern = np.array([[1, 0, -1]])
dy_kern = np.array([[1], [0], [-1]])
print("input image:", IN_PATH, im.shape, im.dtype)


def gen_g(d, sg):
    return cv2.getGaussianKernel(ksize=d, sigma=sg, ktype=cv2.CV_32F)
    # Custom method; but I don't think they want it
    x = np.arange(-(d//2), d//2+1)
    g = np.exp(-(x**2) / (2 * sg**2))
    return g / g.sum()


def gen_G(d, sg):
    g = gen_g(d, sg)
    return np.outer(g, g)


G = gen_G(15, 2)
print("G.shape:", G.shape)

# visualize DoG (x and y)
DxoG = convolve2d(G, dx_kern, mode="same",
                  boundary="fill", fillvalue=0)
# cv2.imwrite(OUT_PATH / "dxog.png", np.abs(DxoG) * 255) # not displaying well; fade this and use plt
plt.axis("off")
plt.imshow(DxoG, cmap="gray")
plt.colorbar()
plt.title("DoG (in x)")
plt.savefig(OUT_PATH / "dxog.png")
plt.close()
print("DxoG.shape:", DxoG.shape)

DyoG = convolve2d(G, dy_kern, mode="same",
                  boundary="fill", fillvalue=0)
plt.axis("off")
plt.imshow(DyoG, cmap="gray")
plt.colorbar()
plt.title("DoG (in y)")
plt.savefig(OUT_PATH / "dyog.png")
plt.close()

# Result 1: convolve im with G first
im_blur = convolve2d(im, G, mode="same",
                     boundary="fill", fillvalue=0)
out_dx = convolve2d(im_blur, dx_kern, mode="same",
                    boundary="fill", fillvalue=0)
cv2.imwrite(OUT_PATH / "cameraman.dx.png", np.abs(out_dx) * 255)


out_dy = convolve2d(im_blur, dy_kern, mode="same",
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
thresh = 0.075
out_gt = (out_g > thresh) * np.ones(out_g.shape)
cv2.imwrite(OUT_PATH / "cameraman.gt.png", out_gt * 255)


# Result 2: convolve im with DoG
out_dxog = convolve2d(im, DxoG, mode="same",
                      boundary="fill", fillvalue=0)
cv2.imwrite(OUT_PATH / "cameraman.dxog.png", np.abs(out_dxog) * 255)


out_dyog = convolve2d(im, DyoG, mode="same",
                      boundary="fill", fillvalue=0)
cv2.imwrite(OUT_PATH / "cameraman.dyog.png", np.abs(out_dyog) * 255)

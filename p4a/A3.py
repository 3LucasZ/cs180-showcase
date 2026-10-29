from scipy.signal import convolve2d
from pathlib import Path

import cv2
import numpy as np

from A2 import computeH
import rect_1.corr
import rect_2.corr


def applyH(pts_2d, H):
    pts_3d = np.vstack([pts_2d, np.ones_like(pts_2d[0, :])])
    out_3d = H@pts_3d
    out_2d = out_3d[:2] / out_3d[2:3]
    return out_2d


def warpImage(im, H, useNN, bbox=None):
    # Q = [xi yi 1]^T = H * (P=im)
    # P = H^-1 Q
    # output bounding box
    in_height, in_width, _ = im.shape
    in_corners = np.array([[0, 0, in_width-1, in_width-1],
                           [0, in_height-1, in_height-1, 0]])
    if bbox is None:
        warped_corners = applyH(in_corners, H)
        u_min = int(np.floor(warped_corners[0].min()))
        u_max = int(np.ceil(warped_corners[0].max()))
        v_min = int(np.floor(warped_corners[1].min()))
        v_max = int(np.ceil(warped_corners[1].max()))
    else:
        u_min, u_max, v_min, v_max = bbox

    # inverse warp
    u, v = np.meshgrid(
        np.arange(u_min, u_max+1),
        np.arange(v_min, v_max+1)
    )
    out_shape = u.shape
    u = u.ravel()
    v = v.ravel()
    out_pts = np.vstack([u, v])
    Hi = np.linalg.inv(H)
    in_pts = applyH(out_pts, Hi)
    # 2 functions: x(u, v), y(u, v)
    x, y = in_pts[0], in_pts[1]
    # mask of valid input coordinates
    ok = (x >= 0) & (x <= in_width-1) & (y >= 0) & (y <= in_height-1)
    x = x[ok]
    y = y[ok]
    # nearest neighbor interpolation
    out = np.zeros((u.size, 3), dtype=im.dtype)
    if useNN:
        xnn = np.rint(x).astype(int)
        ynn = np.rint(y).astype(int)
        # for (u,v) in out: out[u,v] = im[y(u,v),x(u,v)]
        out[ok] = im[ynn, xnn]
    # bilinear interpolation
    else:
        # filter only valid input coordinates
        x0 = np.floor(x).astype(int)
        x1 = np.minimum(x0+1, in_width-1)
        y0 = np.floor(y).astype(int)
        y1 = np.minimum(y0+1, in_height-1)
        im00 = im[y0, x0]
        im01 = im[y0, x1]
        im10 = im[y1, x0]
        im11 = im[y1, x1]
        # broadcast (N,0) bil weights
        dx = (x-x0)[:, None]
        dy = (y-y0)[:, None]
        out[ok] = (
            (1-dx)*(1-dy)*im00 +
            dx*(1-dy)*im10 +
            (1-dx)*dy*im01 +
            dx*dy*im11
        )
    return out.reshape(out_shape[0], out_shape[1], 3), ok.reshape(out_shape), (u_min, v_min)


def warpImageNearestNeighbor(im, H, bbox=None):
    return warpImage(im, H, True, bbox)


def warpImageBilinear(im, H, bbox=None):
    return warpImage(im, H, False, bbox)


DIR = Path(__file__).resolve().parent
np.set_printoptions(precision=2, suppress=True)

# copy sharpen code from 2.1.py


def gen_g(d, sg):
    return cv2.getGaussianKernel(ksize=d, sigma=sg, ktype=cv2.CV_32F)


def gen_G(d, sg):
    g = gen_g(d, sg)
    return np.outer(g, g)


def sharpen(im, sigma, a):
    ksize = 2 * int(3 * sigma) + 1
    G = gen_G(ksize, sigma)
    e = np.zeros_like(G)
    e[G.shape[0] // 2, G.shape[1] // 2] = 1
    unsharp_kern = (1 + a) * e - a * G
    out = np.zeros_like(im, dtype=np.float32)
    for ch in range(3):
        out[:, :, ch] = convolve2d(im[:, :, ch], unsharp_kern, mode="same",
                                   boundary="fill", fillvalue=0)
    return out


# experiments
if __name__ == "__main__":
    im = cv2.imread(DIR / "rect_1" / "src.JPG")
    H = computeH(rect_1.corr.dst_pts, rect_1.corr.src_pts)

    nn, mask, offset = warpImageNearestNeighbor(
        im, H, bbox=(0, 1000, 0, 500))
    cv2.imwrite(DIR / "rect_1" / "dst.nn.JPG", nn)

    print(rect_1.corr.dst_pts)
    print(rect_1.corr.src_pts)
    print("H:", H)
    print("warped shape:", nn.shape)
    print(offset)

    bil, mask, offset = warpImageBilinear(
        im, H, bbox=(0, 1000, 0, 500))
    # cv2.imwrite(DIR / "rect_1" / "dst.bil.JPG", bil)


if __name__ == "__main__":
    im = cv2.imread(DIR / "rect_2" / "src.JPG")
    H = computeH(rect_2.corr.dst_pts, rect_2.corr.src_pts)
    bbox = (0, 1000, 0, 1000)

    nn, _, _ = warpImageBilinear(
        im, H, bbox)
    # cv2.imwrite(DIR / "rect_2" / "dst.nn.JPG", nn)
    # cv2.imwrite(DIR / "rect_2" / "dst.nn.JPG", sharpen(nn, 5, 5))

    bil, _, _ = warpImageBilinear(
        im, H, bbox)
    cv2.imwrite(DIR / "rect_2" / "dst.bil.JPG", bil)
    # cv2.imwrite(DIR / "rect_2" / "dst.bil.JPG", sharpen(bil, 5, 5))

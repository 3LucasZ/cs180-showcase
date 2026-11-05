from pathlib import Path

import cv2
from matplotlib import pyplot as plt
import numpy as np

from B1 import get_anms_corners


def extract_feature_descriptor(im, point):
    # print(point)
    y = point[0]
    x = point[1]
    # 40x40 patch
    patch = im[y-20:y+20, x-20:x+20]
    # 8x8 patch
    patch_b = cv2.GaussianBlur(
        patch, (0, 0), sigmaX=2.5)  # nyquist anti-aliasing
    fd = patch_b[2::5, 2::5]
    # normalize
    fd = (fd - np.mean(fd)) / np.std(fd)
    return fd


if __name__ == "__main__":
    DIR = Path(__file__).resolve().parent
    t = 1
    IN_PATH = DIR / f"mosaic_{t}"
    OUT_PATH = DIR / f"mosaic_{t}"
    im = cv2.imread(IN_PATH / "center.JPG")

    f = 0.25
    im = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # get the 5 best corners
    im_h, corners = get_anms_corners(im_gray)  # sorted by best r
    h = im_h[corners[0], corners[1]]
    sorted_ids = np.argsort(h)
    best_ids = sorted_ids[::-1][:5]  # sorted by best h
    best_corners = corners[:, best_ids]
    print("best_corners", best_corners)

    # visualize their descriptors
    fds = [extract_feature_descriptor(im_gray, corner)
           for corner in best_corners.T]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
              (0, 255, 255), (255, 255, 0)]
    for i in range(5):
        cv2.circle(im, center=(best_corners[1, i], best_corners[0, i]),
                   radius=10, color=colors[i], thickness=-1)
        fd = (fds[i] + 2) / 4 * 255
        fd = fd.astype(np.uint8)
        fd = cv2.cvtColor(fd, cv2.COLOR_GRAY2BGR)
        fd = cv2.copyMakeBorder(
            fd, 1, 1, 1, 1,
            borderType=cv2.BORDER_CONSTANT, value=colors[i]
        )
        fd = cv2.resize(fd, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(OUT_PATH / f"fd_{i}.JPG", fd)
    cv2.imwrite(OUT_PATH / "fd_base.JPG", im)

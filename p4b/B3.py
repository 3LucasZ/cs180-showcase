import colorsys
from pathlib import Path

import cv2
import numpy as np

from B1 import dist2, get_anms_corners
from B2 import color_gen, extract_feature_descriptor


def find_matches(fds1, fds2):
    d2s = dist2(fds1, fds2)
    matches = []
    for i in range(len(fds1)):
        # d = np.sqrt(d2s[i])
        d = d2s[i]
        best_js = np.argsort(d)
        nn1_id = best_js[0]
        nn2_id = best_js[1]
        if d[nn1_id] / d[nn2_id] < 0.6:
            matches.append((i, nn1_id))
    return matches


if __name__ == "__main__":
    DIR = Path(__file__).resolve().parent
    t = 1
    IN_PATH = DIR / f"mosaic_{t}"
    OUT_PATH = DIR / f"mosaic_{t}"
    im1 = cv2.imread(IN_PATH / "1.JPG")
    im2 = cv2.imread(IN_PATH / "2.JPG")

    f = 0.1
    im1 = cv2.resize(im1, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    im2 = cv2.resize(im2, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    _, corners1 = get_anms_corners(im1_gray)
    print("corners1:", corners1.shape)
    _, corners2 = get_anms_corners(im2_gray)

    fds1 = np.array([extract_feature_descriptor(im1_gray, corner).flatten()
                     for corner in corners1.T])
    print("fds1:", fds1.shape)
    fds2 = np.array([extract_feature_descriptor(im2_gray, corner).flatten()
                     for corner in corners2.T])
    matches = find_matches(fds1, fds2)
    print("matches:", len(matches))

    for match in matches:
        i = match[0]
        j = match[1]
        c = color_gen(i)
        cv2.circle(im1, center=(corners1[1, i], corners1[0, i]),
                   radius=10, color=c, thickness=-1)
        cv2.circle(im2, center=(corners2[1, j], corners2[0, j]),
                   radius=10, color=c, thickness=-1)
    cv2.imwrite(OUT_PATH / "1_matching.JPG", im1)
    cv2.imwrite(OUT_PATH / "2_matching.JPG", im2)

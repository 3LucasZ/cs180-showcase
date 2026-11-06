
from B3 import find_matches
from B2 import color_gen, extract_feature_descriptor
from B1 import get_anms_corners
import numpy as np
import cv2
from pathlib import Path

from A import computeH, manual_mosaic2, manual_mosaic3, t_applyH


def ransac(pts1, pts2):
    its = 1000
    t = 5
    best_inliers_len = 0
    best_inliers = None
    for _ in range(its):
        ids = np.random.choice(len(pts1), size=4, replace=False)
        H = computeH(pts2[ids], pts1[ids])
        # print("H.shape", H.shape)
        # print("pts1.shape", pts1.shape)
        pts2_hat = t_applyH(pts1, H)
        errs = np.sum((pts2_hat - pts2) ** 2, axis=1)
        # print("errs:", errs)
        inliers = errs < t
        if inliers.sum() > best_inliers_len:
            best_inliers = inliers
            best_inliers_len = inliers.sum()
    return best_inliers


def get_corrs(im1, im2, verbose=True):
    if verbose:
        print("CORRELATING...")
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    _, corners1 = get_anms_corners(im1_gray, verbose=verbose)
    # print("corners1:", corners1.shape)
    _, corners2 = get_anms_corners(im2_gray, verbose=verbose)

    fds1 = np.array([extract_feature_descriptor(im1_gray, corner).flatten()
                     for corner in corners1.T])
    # print("fds1:", fds1.shape)
    fds2 = np.array([extract_feature_descriptor(im2_gray, corner).flatten()
                     for corner in corners2.T])
    matches = find_matches(fds1, fds2, verbose=verbose)
    # print("matches:", len(matches))
    # [::-1] to convert from (y,x) -> (x,y), smh so stupid
    pts1 = np.array([corners1[:, i][::-1] for i, j in matches])
    pts2 = np.array([corners2[:, j][::-1] for i, j in matches])
    # print("pts1:", pts1.shape)
    inliers = ransac(pts1, pts2)
    # print(inliers)
    pts1 = pts1[inliers]
    pts2 = pts2[inliers]
    # print("pts1:", pts1)
    if verbose:
        print("RANSAC inliers:", len(pts1))
    return pts1, pts2


def auto_mosaic2(im1, im2):
    # NOTE: left, center, with center being the frame of reference
    pts1, pts2 = get_corrs(im1, im2)
    return manual_mosaic2(im1, im2, pts1, pts2)


def auto_mosaic3(im1, im2, im3):
    # return auto_mosaic2(im1, auto_mosaic2(im3, im2))
    pts1_pts2 = get_corrs(im1, im2)
    pts2_pts3 = get_corrs(im2, im3)
    return manual_mosaic3(
        im1, im2, im3, pts1_pts2[0], pts1_pts2[1], pts2_pts3[0], pts2_pts3[1])


if __name__ == "__main__":
    DIR = Path(__file__).resolve().parent
    t = 1
    IN_PATH = DIR / f"mosaic_{t}"
    OUT_PATH = DIR / f"mosaic_{t}"
    im1 = cv2.imread(IN_PATH / "1.JPG")
    im2 = cv2.imread(IN_PATH / "2.JPG")
    im3 = cv2.imread(IN_PATH / "3.JPG")

    f = 0.25
    im1 = cv2.resize(im1, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    im2 = cv2.resize(im2, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    im3 = cv2.resize(im3, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

    vis_ransac = True
    if vis_ransac:
        pts1, pts2 = get_corrs(im1, im2, verbose=False)
        for i in range(len(pts1)):
            c = color_gen(i)
            cv2.circle(im1, center=pts1[i],
                       radius=5, color=c, thickness=-1)
            cv2.circle(im2, center=pts2[i],
                       radius=5, color=c, thickness=-1)
        cv2.imwrite(OUT_PATH / "ransac_1.JPG", im1)
        cv2.imwrite(OUT_PATH / "ransac_2.JPG", im2)

    total = auto_mosaic3(im1, im2, im3)
    cv2.imwrite(OUT_PATH / "auto_mosaic.jpg", total)

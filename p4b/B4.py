
from B3 import find_matches
from B2 import color_gen, extract_feature_descriptor
from B1 import get_anms_corners
import numpy as np
import cv2
from pathlib import Path

# COPY FROM A


def applyH(pts_2d, H):
    pts_2d = np.transpose(pts_2d, (1, 0))
    pts_3d = np.vstack([pts_2d, np.ones_like(pts_2d[0, :])])
    # print("pts_3d.shape", pts_3d.shape)
    out_3d = H@pts_3d
    out_2d = out_3d[:2] / out_3d[2:3]
    out_2d = np.transpose(out_2d, (1, 0))
    return out_2d


def computeH(ref, to_project):
    # find H
    # H = 3x3 matrix with 8DoF since lower right corner is 1
    # such that p' = Hp
    # p = the points we want to project (3 x N) where N = number of correspondences
    # p' = the points we want to project should look like ref after projection (3 x N)
    # expanding all the terms
    # p1 = linear in (h1, ..., h8) with constants (p1, p2, q1, q2) used
    ref = np.astype(ref, np.float64)
    to_project = np.astype(to_project, np.float64)
    A = []
    b = []
    corr = np.hstack((to_project, ref))
    for p1, p2, q1, q2 in corr:
        A.append([p1, p2, 1, 0, 0, 0, -q1*p1, -q1*p2])
        A.append([0, 0, 0, p1, p2, 1, -q2*p1, -q2*p2])
        b.extend([q1, q2])
    x, err, _, _ = np.linalg.lstsq(A, b, rcond=None)
    H = np.append(x, 1)
    # print("Error:", err)
    return H.reshape(3, 3)


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
        pts2_hat = applyH(pts1, H)
        errs = np.sum((pts2_hat - pts2) ** 2, axis=1)
        # print("errs:", errs)
        inliers = errs < t
        if inliers.sum() > best_inliers_len:
            best_inliers = inliers
            best_inliers_len = inliers.sum()
    return best_inliers


def get_corrs(im1, im2):
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    _, corners1 = get_anms_corners(im1_gray)
    # print("corners1:", corners1.shape)
    _, corners2 = get_anms_corners(im2_gray)

    fds1 = np.array([extract_feature_descriptor(im1_gray, corner).flatten()
                     for corner in corners1.T])
    # print("fds1:", fds1.shape)
    fds2 = np.array([extract_feature_descriptor(im2_gray, corner).flatten()
                     for corner in corners2.T])
    matches = find_matches(fds1, fds2)
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
    # print("inliers:", len(pts1))
    return pts1, pts2


if __name__ == "__main__":
    DIR = Path(__file__).resolve().parent
    t = 1
    IN_PATH = DIR / f"mosaic_{t}"
    OUT_PATH = DIR / f"mosaic_{t}"
    im1 = cv2.imread(IN_PATH / "1.JPG")
    im2 = cv2.imread(IN_PATH / "2.JPG")
    im3 = cv2.imread(IN_PATH / "3.JPG")

    f = 0.1
    im1 = cv2.resize(im1, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    im2 = cv2.resize(im2, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    pts1, pts2 = get_corrs(im1, im2)
    if False:
        for i in range(len(pts1)):
            c = color_gen(i)
            cv2.circle(im1, center=pts1[i],
                       radius=5, color=c, thickness=-1)
            cv2.circle(im2, center=pts2[i],
                       radius=5, color=c, thickness=-1)
        cv2.imwrite(OUT_PATH / "ransac_1.JPG", im1)
        cv2.imwrite(OUT_PATH / "ransac_2.JPG", im2)

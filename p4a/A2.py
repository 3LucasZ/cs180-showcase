import numpy as np

import mosaic_1.corr


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
    print("Error:", err)
    return H.reshape(3, 3)


if __name__ == "__main__":
    H = computeH(mosaic_1.corr.center_right_pts, mosaic_1.corr.right_pts)
    print(H)

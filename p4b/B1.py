
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import corner_harris, peak_local_max

# GIVEN


def get_harris_corners(im, edge_discard=20, verbose=True):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    """

    assert edge_discard >= 20

    # find harris corners
    h = corner_harris(im, method='eps', sigma=1)
    coords = peak_local_max(h, min_distance=1)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask].T
    if verbose:
        print("Harris corners:", coords.shape)
    return h, coords


# GIVEN


def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points.

    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them.  Both matrices must be of
    the same column dimension.  If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns.  The
    I, Jth entry is the  squared distance from the Ith row of X to the
    Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """

    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert dimx == dimc, 'Data dimension does not match dimension of centers'

    return (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T + \
        np.ones((ndata, 1)) * np.sum((c**2).T, axis=0) - \
        2 * np.inner(x, c)

# ME AND MYSELF


def get_anms_corners(im, verbose=True):
    capacity = 2500
    im_h, corners = get_harris_corners(im, verbose=verbose)
    h = im_h[corners[0], corners[1]]
    sorted_ids = np.argsort(h)
    best_ids = sorted_ids[::-1][:capacity]  # sorted by largest h
    best_corners = corners[:, best_ids]
    corners = best_corners

    k = 250
    cornersT = corners.T
    h = im_h[corners[0], corners[1]]
    r = np.zeros(len(cornersT))
    for i in range(len(cornersT)):
        js = 0.9 * h > h[i]
        if np.any(js):
            d2s = dist2(cornersT[i:i+1], cornersT[js])
            r[i] = np.min(d2s)
        else:
            r[i] = np.inf
    sorted = np.argsort(r)
    best = sorted[::-1][:k]
    if verbose:
        print("ANMS corners:", k)
    return im_h, corners[:, best]


if __name__ == "__main__":
    DIR = Path(__file__).resolve().parent
    t = 1
    IN_PATH = DIR / f"mosaic_{t}"
    OUT_PATH = DIR / f"mosaic_{t}"
    im = cv2.imread(IN_PATH / "2.JPG")

    # downscale everything because massive images from iphone are hard to process
    f = 0.25
    # print("left.shape", left.shape)
    im = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    # print("left.shape", left.shape)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # harris
    if True:
        im_h, corners = get_harris_corners(im_gray)
        # print(corners.shape)
        # print(corners)
        im1 = im.copy()
        for point in np.transpose(corners):
            cv2.circle(im1, center=(int(point[1]), int(point[0])),
                       radius=1, color=(0, 0, 255), thickness=-1)
        cv2.imwrite(OUT_PATH / "corners.JPG", im1)
    if True:
        # topk
        h = im_h[corners[0], corners[1]]
        sorted_ids = np.argsort(h)
        best_ids = sorted_ids[::-1][:250]  # sorted by largest h
        best_corners = corners[:, best_ids]
        im3 = im.copy()
        for point in np.transpose(best_corners):
            cv2.circle(im3, center=(int(point[1]), int(point[0])),
                       radius=5, color=(0, 0, 255), thickness=-1)
        cv2.imwrite(OUT_PATH / "corners_besth.JPG", im3)
    if True:
        # anms
        h, corners = get_anms_corners(im_gray)
        im2 = im.copy()
        for point in np.transpose(corners):
            cv2.circle(im2, center=(int(point[1]), int(point[0])),
                       radius=5, color=(0, 0, 255), thickness=-1)
        cv2.imwrite(OUT_PATH / "corners_ANMS.JPG", im2)

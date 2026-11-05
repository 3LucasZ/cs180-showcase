from pathlib import Path

import cv2


if __name__ == "__main__":
    DIR = Path(__file__).resolve().parent
    t = 1
    IN_PATH = DIR / f"mosaic_{t}"
    OUT_PATH = DIR / f"mosaic_{t}"
    im1 = cv2.imread(IN_PATH / "left.JPG")
    im2 = cv2.imread(IN_PATH / "center.JPG")

    f = 0.25
    im1 = cv2.resize(im1, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    im2 = cv2.resize(im2, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

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

from pathlib import Path

import cv2
import numpy as np

from B4 import auto_mosaic2, auto_mosaic3

if __name__ == "__main__" and False:
    DIR = Path(__file__).resolve().parent
    IN_PATH = DIR / f"panorama"
    OUT_PATH = DIR / f"panorama"
    t = 1
    im1 = cv2.imread(IN_PATH / f"{t}.JPG")
    im2 = cv2.imread(IN_PATH / f"{t+1}.JPG")
    im3 = cv2.imread(IN_PATH / f"{t+2}.JPG")

    f = 0.25
    im1 = cv2.resize(im1, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    im2 = cv2.resize(im2, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    im3 = cv2.resize(im3, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

    if False:
        total = auto_mosaic3(im1, im2, im3)
        cv2.imwrite(OUT_PATH / "auto_mosaic.jpg", total)

# not good! let's do better by doing cylindrical warp!
# warp function is not part of core project, thus used help of AI


def cylindrical_warp(image, focal):
    height, width = image.shape[:2]
    cx, cy = (width - 1) / 2, (height - 1) / 2
    half_width = focal * np.arctan(cx / focal)
    out_width = max(1, 2 * int(np.floor(half_width)) + 1)
    out_cx = (out_width - 1) / 2

    u, v = np.meshgrid(
        np.arange(out_width, dtype=np.float32) - out_cx,
        np.arange(height, dtype=np.float32) - cy,
    )
    angle = u / focal
    map_x = focal * np.tan(angle) + cx
    map_y = v / np.cos(angle) + cy

    valid = ((map_x >= 0) & (map_x <= width - 1) &
             (map_y >= 0) & (map_y <= height - 1))
    warped = cv2.remap(image, map_x.astype(np.float32),
                       map_y.astype(np.float32), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    warped[~valid] = 0
    # warped image, mask, center of image
    return warped, valid.astype(np.uint8), out_cx


'''
exiftool -FocalLength -FocalLengthIn35mmFormat -ImageWidth -ImageHeight p4b/EXTRA/1.JPG
Focal Length                    : 1.6 mm
Focal Length In 35mm Format     : 13 mm
Image Width                     : 4032
Image Height                    : 3024
13/43.27 * sqrt(4032^2 + 3024^2) =  1514
'''

if __name__ == "__main__" and False:
    DIR = Path(__file__).resolve().parent
    IN_PATH = DIR / f"panorama"
    OUT_PATH = DIR / f"panorama"
    t = 1
    im1 = cv2.imread(IN_PATH / f"{t}.JPG")
    im2 = cv2.imread(IN_PATH / f"{t+1}.JPG")
    im3 = cv2.imread(IN_PATH / f"{t+2}.JPG")

    f = 0.25
    im1 = cv2.resize(im1, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    im2 = cv2.resize(im2, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    im3 = cv2.resize(im3, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

    im1warp, _, _ = cylindrical_warp(im1, 1514 * f)
    im2warp, _, _ = cylindrical_warp(im2, 1514 * f)
    im3warp, _, _ = cylindrical_warp(im3, 1514 * f)

    cv2.imwrite(OUT_PATH / "1_warp.jpg", im1warp)
    cv2.imwrite(OUT_PATH / "2_warp.jpg", im2warp)
    cv2.imwrite(OUT_PATH / "3_warp.jpg", im3warp)
    cyl_mosaic3 = auto_mosaic3(im1warp, im2warp, im3warp)
    cv2.imwrite(OUT_PATH / "cyl_mosaic3.jpg", cyl_mosaic3)


def crop(im):
    h, w, _ = im.shape
    return im[h//4:3*h//4, :]


if __name__ == "__main__" and False:
    DIR = Path(__file__).resolve().parent
    IN_PATH = DIR / f"panorama"
    OUT_PATH = DIR / f"panorama"
    t = 1
    im1 = cv2.imread(IN_PATH / f"{t}.JPG")
    im2 = cv2.imread(IN_PATH / f"{t+1}.JPG")
    im3 = cv2.imread(IN_PATH / f"{t+2}.JPG")

    f = 0.25
    im1 = cv2.resize(im1, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    im2 = cv2.resize(im2, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    im3 = cv2.resize(im3, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

    im1warp, _, _ = cylindrical_warp(im1, 1514 * f)
    im2warp, _, _ = cylindrical_warp(im2, 1514 * f)
    im3warp, _, _ = cylindrical_warp(im3, 1514 * f)
    im1warp = crop(im1warp)
    im2warp = crop(im2warp)
    im3warp = crop(im3warp)

    cyl_mosaic3 = auto_mosaic3(im1warp, im2warp, im3warp)
    cv2.imwrite(OUT_PATH / "crop_cyl_mosaic3.jpg", cyl_mosaic3)

# do all the images now!?

if __name__ == "__main__" and True:
    DIR = Path(__file__).resolve().parent
    IN_PATH = DIR / f"panorama"
    OUT_PATH = DIR / f"panorama"
    N = 10
    ims = [cv2.imread(IN_PATH / f"{i+1}.JPG") for i in range(N)]
    f = 0.5
    ims = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
           for im in ims]
    focal = 1514 * f
    for i in range(N):
        warp, mask, _ = cylindrical_warp(ims[i], focal)
        ims[i] = crop(warp)
    panorama = ims[0]
    for i in range(N-1):
        panorama = auto_mosaic2(panorama, ims[i+1])
        cv2.imwrite(OUT_PATH / f"crop_cyl_mosaicN_{i}.jpg", panorama)
        print(panorama.shape)
    cv2.imwrite(OUT_PATH / "crop_cyl_mosaicN.jpg", panorama)

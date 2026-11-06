'''
exiftool -FocalLength -FocalLengthIn35mmFormat -ImageWidth -ImageHeight panorama2/1.jpg
Focal Length                    : 18.0 mm
Focal Length In 35mm Format     : 27 mm
Image Width                     : 1200
Image Height                    : 795
'''
import numpy as np

from B5_1 import cylindrical_warp
import cv2
from pathlib import Path

from B4 import auto_mosaic2, get_corrs
from A import stitch
focal = 27 * np.hypot(1200, 795) / np.hypot(36, 24)

# recursion is too slow
if __name__ == "__main__" and False:
    DIR = Path(__file__).resolve().parent
    IN_PATH = DIR / f"panorama2"
    OUT_PATH = DIR / f"panorama2"
    N = 12
    ims = [cv2.imread(IN_PATH / f"{i+1}.jpg") for i in range(N)]
    f = 1
    ims = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
           for im in ims]
    focal = 1514 * f
    for i in range(N):
        warp, mask, _ = cylindrical_warp(ims[i], focal)
        ims[i] = warp
    panorama = ims[0]
    for i in range(N-1):
        panorama = auto_mosaic2(panorama, ims[i+1])
        # cv2.imwrite(OUT_PATH / f"crop_cyl_mosaicN_{i}.jpg", panorama)
        # print(panorama.shape)
    cv2.imwrite(OUT_PATH / "crop_cyl_mosaicN.jpg", panorama)


def auto_mosaic_translate(ims, final_crop=False, close_loop=False):
    tot_dx, tot_dy = 0.0, 0.0
    dataset = [(ims[0], (0, 0))]
    for i in range(len(ims)-1):
        im1 = ims[i]
        im2 = ims[i+1]
        pts1, pts2 = get_corrs(im1, im2)
        # translation model instead of homography model
        dx, dy = np.median(pts1 - pts2, axis=0)
        tot_dx, tot_dy = tot_dx + dx, tot_dy + dy
        dataset.append((ims[i+1], (round(tot_dx), round(tot_dy))))
    x_min = min(origin[0] for im, origin in dataset)
    y_min = min(origin[1] for im, origin in dataset)
    x_max = max(origin[0] + im.shape[1] for im, origin in dataset)
    y_max = max(origin[1] + im.shape[0] for im, origin in dataset)

    h_mosaic = y_max - y_min
    w_mosaic = x_max - x_min

    frames = []
    masks = []

    # turn each image into a frame
    for im, (x, y) in dataset:
        h, w = im.shape[:2]
        y_off = y - y_min
        x_off = x - x_min
        frame = np.zeros((h_mosaic, w_mosaic, 3), dtype=im.dtype)
        frame_mask = np.zeros((h_mosaic, w_mosaic), dtype=bool)
        frame[y_off:y_off+h, x_off:x_off+w] = im
        frame_mask[y_off:y_off+h, x_off:x_off+w] = 1
        frames.append(frame)
        masks.append(frame_mask)

    total = frames[0]
    total_mask = masks[0]
    for i in range(len(frames)-1):
        total = stitch(total, frames[i+1], total_mask, masks[i+1])
        total_mask = total_mask | masks[i+1]

    # final crop
    if final_crop:
        crop_y_min = max(y for im, (x, y) in dataset) - y_min
        crop_y_max = min(y + im.shape[0] for im, (x, y) in dataset) - y_min
        total = total[crop_y_min:crop_y_max, :]

    # final x crop to close the loop
    if close_loop:
        pts1, pts2 = get_corrs(ims[0], ims[-1])
        last_x = dataset[-1][1][0]
        width = last_x + np.median(pts2[:, 0] - pts1[:, 0])
        start = -x_min
        end = start + round(width)
        total = total[:, start:end]  # horizontal crop

    return total


def crop(im):
    h, w, _ = im.shape
    r = 12
    return im[h//r:(r-1)*h//r, :]


# VERY GOOD RESULTS PRAISE THE LORD
if __name__ == "__main__" and False:
    DIR = Path(__file__).resolve().parent
    IN_PATH = DIR / f"panorama2"
    OUT_PATH = DIR / f"panorama2"
    N = 12
    ims = [cv2.imread(IN_PATH / f"{i+1}.jpg") for i in range(N)]
    for i in range(N):
        warp, mask, _ = cylindrical_warp(ims[i], focal)
        ims[i] = warp
    panorama = auto_mosaic_translate(ims)
    cv2.imwrite(OUT_PATH / "mosaic.jpg", panorama)

if __name__ == "__main__" and False:
    DIR = Path(__file__).resolve().parent
    IN_PATH = DIR / f"panorama2"
    OUT_PATH = DIR / f"panorama2"
    N = 12
    ims = [cv2.imread(IN_PATH / f"{i+1}.jpg") for i in range(N)]
    for i in range(N):
        warp, mask, _ = cylindrical_warp(ims[i], focal)
        ims[i] = crop(warp)
    panorama = auto_mosaic_translate(ims)
    cv2.imwrite(OUT_PATH / "mosaic_crop.jpg", panorama)

if __name__ == "__main__" and False:
    DIR = Path(__file__).resolve().parent
    IN_PATH = DIR / f"panorama2"
    OUT_PATH = DIR / f"panorama2"
    N = 12
    ims = [cv2.imread(IN_PATH / f"{i+1}.jpg") for i in range(N)]
    for i in range(N):
        warp, mask, _ = cylindrical_warp(ims[i], focal)
        ims[i] = crop(warp)
    panorama = auto_mosaic_translate(ims, final_crop=True)
    cv2.imwrite(OUT_PATH / "mosaic_crop_crop.jpg", panorama)

# time to close the loop!
if __name__ == "__main__" and False:
    DIR = Path(__file__).resolve().parent
    IN_PATH = DIR / f"panorama2"
    OUT_PATH = DIR / f"panorama2"
    N = 12
    ims = [cv2.imread(IN_PATH / f"{i+1}.jpg") for i in range(N)]
    for i in range(N):
        warp, mask, _ = cylindrical_warp(ims[i], focal)
        ims[i] = crop(warp)
    panorama = auto_mosaic_translate(ims, final_crop=True, close_loop=True)
    cv2.imwrite(OUT_PATH / "mosaic_crop_crop_loop.jpg", panorama)

'''
exiftool -FocalLength -FocalLengthIn35mmFormat -ImageWidth -ImageHeight panorama2/1.jpg
Focal Length                    : 16.0 mm
Focal Length In 35mm Format     : 24 mm
Image Width                     : 1300
Image Height                    : 867
'''
focal = 867
if __name__ == "__main__" and False:
    DIR = Path(__file__).resolve().parent
    IN_PATH = DIR / f"panorama3"
    OUT_PATH = DIR / f"panorama3"
    N = 14  # most of the images after 14 are useless and bad
    ims = [cv2.imread(IN_PATH / f"{i+1}.JPG") for i in range(N)]
    for i in range(N):
        warp, mask, _ = cylindrical_warp(ims[i], focal)
        ims[i] = crop(warp)
    panorama = auto_mosaic_translate(ims, final_crop=True, close_loop=True)
    cv2.imwrite(OUT_PATH / "mosaic.jpg", panorama)
'''
Focal Length                    : 13.1 mm
Focal Length In 35mm Format     : 74 mm
Image Width                     : 1600
Image Height                    : 1200
'''
focal = 3421
if __name__ == "__main__" and True:
    DIR = Path(__file__).resolve().parent
    IN_PATH = DIR / f"panorama4"
    OUT_PATH = DIR / f"panorama4"
    N = 12  # last 2 images are kinda faulty.
    ims = [cv2.imread(IN_PATH / f"{i+1}.JPG") for i in range(N)]
    for i in range(N):
        warp, mask, _ = cylindrical_warp(ims[i], focal)
        ims[i] = crop(warp)
    # loop closure is not working here, faulty dataset
    panorama = auto_mosaic_translate(ims, final_crop=False, close_loop=False)
    cv2.imwrite(OUT_PATH / "mosaic.jpg", panorama)

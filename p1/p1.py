import numpy as np
import cv2
from pathlib import Path

# consts

window_sz = 8

# for placing the 3 shifted images on top of each other


def pad(im):
    p = im.shape[0] // 10
    return cv2.copyMakeBorder(
        im,
        p, p,
        p, p,
        cv2.BORDER_CONSTANT,
        value=0
    )

# for removing the frame around the images before alignment


def crop(im):
    p = im.shape[0] // 10
    return im[p:-p, p:-p]


def downsample2(im):
    # crop image to be even size before downsampling by 2
    if (im.shape[0] % 2 == 1):
        im = im[1:, :]
    if (im.shape[1] % 2 == 1):
        im = im[:, 1:]
    return (im[0::2, 0::2] + im[0::2, 1::2] + im[1::2, 0::2] + im[1::2, 1::2]) / 4.0


def norm(u):
    u = u.flatten()
    u = u - np.mean(u)
    return u/np.linalg.norm(u)


# similarity score, higher is better
def ncc(u, v):
    return np.dot(norm(u), norm(v))


# difference score, lower is better
def ed(u, v):
    u = u.flatten()
    v = v.flatten()
    return np.sqrt(np.sum((u-v) ** 2))


def apply_shift(img1, img2, shift):
    (dy, dx) = shift
    (h, w) = img1.shape
    # crop both images to same size on opposite ends to simulate a shift
    dy *= -1
    dx *= -1
    img1 = img1[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)]
    img2 = img2[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
    return img1, img2

# (y, x) is the current best alignment vector for u


def find_shift_pyramid(u, v):
    if (u.shape[0] < 64):
        return find_shift(u, v)
    # recurse: solve downsampled version first and use that result to shift first
    y, x = find_shift_pyramid(downsample2(u), downsample2(v))
    y, x = y * 2, x * 2
    return find_shift(u, v, y, x)


def find_shift(u, v, y=0, x=0):
    # print("find_shift:", u.shape)
    top_score = None
    new_dy = 0
    new_dx = 0
    for try_dy in range(-window_sz, window_sz+1):
        for try_dx in range(-window_sz, window_sz+1):
            # Rolling wraps the image weirdly messing up results
            # up = np.roll(u, shift=(dy, dx), axis=(0, 1))
            up, vp = apply_shift(u, v, (y+try_dy, x+try_dx))
            # print(up.shape, vp.shape)
            # score = ed(up, vp)
            score = ncc(up, vp)
            # print(dy, dx, score)
            if top_score is None or score > top_score:
                top_score = score
                new_dy = try_dy
                new_dx = try_dx
    # print(shift, top_score)
    return y+new_dy, x+new_dx


def align(u, v):
    shift = find_shift_pyramid(u, v)
    # u = pad(u)
    return np.roll(u, shift=shift, axis=(0, 1))


def main(IN_PATH, OUT_PATH):

    im = cv2.imread(IN_PATH, cv2.IMREAD_GRAYSCALE)
    print("input image:", IN_PATH, im.shape, im.dtype)
    im = im.astype(np.float32)/255
    height = np.floor(im.shape[0] / 3.0).astype(np.uint)

    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]
    # pad(b)
    # pad(g)
    # pad(r)

    # align the images
    # functions that might be useful for aligning the images include:
    # np.roll, np.sum, sk.transform.rescale (for multiscale)
    ag = align(g, b)
    ar = align(r, b)
    # create a color image

    im_out = np.dstack([b, ag, ar])
    im_out = (im_out * 255).astype(np.uint8)

    # save the image
    cv2.imwrite(OUT_PATH, im_out)
    # display the image
    # cv2.imshow('Window', im_out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


DIR = Path(__file__).resolve().parent
IN_DIR = DIR / "in"
OUT_DIR = DIR / "out"
for IN_PATH in IN_DIR.iterdir():
    if IN_PATH.suffix != ".jpg" and IN_PATH.name != "emir.tif":
        continue
    OUT_PATH = (OUT_DIR / IN_PATH.name).with_suffix(".jpg")
    main(IN_PATH, OUT_PATH)

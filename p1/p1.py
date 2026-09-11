import numpy as np
import cv2
from pathlib import Path

# consts

window_sz = 2

# for placing the 3 shifted images on top of each other


def pad(im, p=15):
    p = im.shape[0] // p
    return cv2.copyMakeBorder(
        im,
        p, p,
        p, p,
        cv2.BORDER_CONSTANT,
        value=1
    )

# for removing the frame around the images before alignment


def crop(im, p=15):
    p = im.shape[0] // p
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


def apply_sobel(u):
    sx = cv2.Sobel(u, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(u, cv2.CV_64F, 0, 1, ksize=3)
    up = np.sqrt(sx**2 + sy**2)
    return up

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
            up = apply_sobel(up)
            vp = apply_sobel(vp)
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


def align(u, v, single_scale):
    if (single_scale):
        shift = find_shift(u, v)
    else:
        # crop before finding shift to remove borders
        shift = find_shift_pyramid(crop(u), crop(v))
    # roll on a pad to simulate a padded shift
    return shift, np.roll(pad(u), shift=shift, axis=(0, 1))


def main(IN_PATH, OUT_PATH, single_scale):

    im = cv2.imread(IN_PATH, cv2.IMREAD_GRAYSCALE)

    # im = cv2.resize(im, (400, 1200), interpolation=cv2.INTER_AREA)
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
    gshift, ag = align(g, b, single_scale)
    rshift, ar = align(r, b, single_scale)
    print("green offset:", gshift)
    print("red offset:", rshift)
    b = pad(b)
    ag = crop(ag)
    ar = crop(ar)
    b = crop(b)
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

# single scale naiive
if False:
    IN_DIR = IN_DIR / f"sample{1}"
    OUT_DIR = OUT_DIR / f"single-scale"
    for IN_PATH in IN_DIR.iterdir():
        OUT_PATH = (OUT_DIR / IN_PATH.name).with_suffix(".jpg")
        main(IN_PATH, OUT_PATH, single_scale=True)

if False:
    IN_PATH = IN_DIR / f"sample{2}" / "emir.tif"
    OUT_PATH = OUT_DIR / f"test" / "bademir.jpg"
    main(IN_PATH, OUT_PATH, False)
# Final product. Try sample_id = 1, 2, 3 for different sample sets.
if True:
    sample_id = 3
    IN_DIR = IN_DIR / f"sample{sample_id}"
    OUT_DIR = OUT_DIR / f"sample{sample_id}"
    for IN_PATH in sorted(IN_DIR.iterdir()):
        # if IN_PATH.suffix != ".jpg" and IN_PATH.name != "emir.tif":
        #     continue
        OUT_PATH = (OUT_DIR / IN_PATH.name).with_suffix(".jpg")
        main(IN_PATH, OUT_PATH, single_scale=False)

# Intermediate product to help me make my website.
if False:
    IN_PATH = IN_DIR / "sample1" / "monastery.jpg"
    OUT_PATH = OUT_DIR / "test" / "no_align.jpg"
    im = cv2.imread(IN_PATH, cv2.IMREAD_GRAYSCALE)
    height = np.floor(im.shape[0] / 3.0).astype(np.uint)
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]
    im_out = np.dstack([b, g, r])
    cv2.imwrite(OUT_PATH, im_out)

if False:
    IN_PATH = IN_DIR / "sample2" / "emir.tif"
    OUT_PATH = OUT_DIR / "test" / "sobel.jpg"
    im = cv2.imread(IN_PATH, cv2.IMREAD_GRAYSCALE)
    height = np.floor(im.shape[0] / 3.0).astype(np.uint)
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]
    b = crop(b)
    g = crop(g)
    r = crop(r)
    b = apply_sobel(b)
    g = apply_sobel(g)
    r = apply_sobel(r)
    im_out = np.hstack([b, g, r])
    cv2.imwrite(OUT_PATH, im_out)

# NOTE: IM A CS180 STUDENT NOT CS280 SO BELLS + WHISTLES ARE OPTIONAL.
# got external help for this section.

from pathlib import Path

import cv2
import numpy as np


def autocrop_frame(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gx = np.abs(cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3))
    h, w = im.shape
    vs = gx.mean(axis=0)
    hs = gy.mean(axis=1)
    sx = int(w * 0.2)
    sy = int(h * 0.2)
    l = np.argmax(vs[:sx])
    r = w - sx + np.argmax(vs[w-sx:])
    t = np.argmax(hs[:sy])
    b = h - sy + np.argmax(hs[h-sy:])
    p = 10
    l += w//p
    r -= w//p
    t += h//p
    b -= h//p
    return t, b, l, r


def expand_chs(im, disjoint=True):
    if disjoint:
        chs = cv2.split(im)
        new_chs = []
        for ch in chs:
            new_chs.append(expand_ch(ch))
        return cv2.merge(new_chs)
    else:
        return expand_ch(im)


def expand_ch(ch):
    a = np.percentile(ch, 3)
    b = np.percentile(ch, 97)
    ch = (ch - a) / (b - a)
    return np.clip(ch, 0, 1)


def awb(im, useWhite=True):
    if useWhite:
        white = np.percentile(im, 99, axis=(0, 1))
        scale = white.max() / (white + 1e-8)
        return np.clip(im * scale, 0, 1)
    # else, use gray as the norm
    else:
        mean_rgb = im.mean(axis=(0, 1))
        gray = mean_rgb.mean()
        scale = gray / (mean_rgb + 1e-8)
        return np.clip(im * scale, 0, 1)


def gmma(im, g=0.8):
    return np.clip(im, 0, 1) ** g


def claher(im, clipLimit=1.2):
    im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clipLimit,
        tileGridSize=(8, 8)
    )
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    res = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return res.astype(np.float32) / 255


def unsharp(im, sigma=1.5, amount=0.7):
    blurred = cv2.GaussianBlur(im, (0, 0), sigma)
    sharpened = cv2.addWeighted(
        im, 1 + amount,
        blurred, -amount,
        0
    )
    return sharpened


def sat(im, f=1.1):
    im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= f
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    out = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2BGR
    )
    return out.astype(np.float32) / 255


def vibrance(im, amount=0.20):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    boost = amount * (1.0 - s)
    s = s * (1.0 + boost)
    hsv[:, :, 1] = np.clip(s, 0, 1)
    return cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )


def color_transfer(src, ref):
    def match_channel(src, ref):
        src_values, src_indices, src_counts = np.unique(
            src.ravel(),
            return_inverse=True,
            return_counts=True
        )
        ref_values, ref_counts = np.unique(
            ref.ravel(),
            return_counts=True
        )
        src_cdf = np.cumsum(src_counts).astype(np.float64)
        src_cdf /= src_cdf[-1]
        ref_cdf = np.cumsum(ref_counts).astype(np.float64)
        ref_cdf /= ref_cdf[-1]
        matched_values = np.interp(
            src_cdf,
            ref_cdf,
            ref_values
        )
        return matched_values[src_indices].reshape(src.shape)

    res = np.zeros_like(src, dtype=np.float32)
    for c in range(3):
        res[:, :, c] = match_channel(
            src[:, :, c],
            ref[:, :, c]
        )
    return np.clip(res, 0, 1)


def lab_color_transfer(src, ref):
    src = (src * 255).astype(np.uint8)
    ref = (ref * 255).astype(np.uint8)
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
    src_lab = src_lab.astype(np.float32) / 255.0
    ref_lab = ref_lab.astype(np.float32) / 255.0
    res = color_transfer(
        src_lab,
        ref_lab
    )
    res = (res * 255).astype(np.uint8)
    return (
        cv2.cvtColor(res, cv2.COLOR_LAB2BGR)
        .astype(np.float32) / 255.0
    )


DIR = Path(__file__).resolve().parent
IN_PATH = DIR / "out" / "sample3" / "mountain.jpg"
OUT_PATH = DIR / "out" / "test"
im = cv2.imread(IN_PATH)
print("input image:", IN_PATH, im.shape, im.dtype)

# cvt to floats
im = im.astype(np.float32)/255

# auto crop
if True:
    t, b, l, r = autocrop_frame(im)
    im = im[t:b, l:r]
    im_out = (im*255).astype(np.uint8)
    cv2.imwrite(OUT_PATH / "crop5.jpg", im_out)


# expand channels
if True:
    im = expand_chs(im, disjoint=True)
    im_out = (im*255).astype(np.uint8)
    # cv2.imwrite(OUT_PATH / "expand_ch.jpg", im_out)

# AWB
if False:
    im = awb(im, useWhite=True)
    im_out = (im*255).astype(np.uint8)
    # cv2.imwrite(OUT_PATH / "awb.jpg", im_out)

# add detail
if True:
    im = claher(im, clipLimit=3)
    # im = unsharp_mask(im)
    im_out = (im*255).astype(np.uint8)
    # cv2.imwrite(OUT_PATH / "sharp.jpg", im_out)

# if True:
#     im = vibrance(im, 0.75)
#     im_out = (im*255).astype(np.uint)
#     cv2.imwrite(OUT_PATH / "vibrant.jpg", im_out)


# denoise glass imperfections
# if True:
#     im = cv2.fastNlMeansDenoisingColored(
#         im,
#         None,
#         h=5,
#         hColor=5,
#         templateWindowSize=7,
#         searchWindowSize=21
#     )
#     im = cv2.medianBlur(im, 3)
#     cv2.imwrite(OUT_PATH / "denoised.jpg", im)


# # im = expand_chs(im)
# im = expand_chs_disjoint(im)
# # im = awb(im)
# # im = awb2(im)


# # im = sat(im, f=1.5)
# # im = vibrance(im, 1)
# # im = gmma(im)

# Color transfer:
if True:
    im2 = cv2.imread(DIR / "out" / "test" /
                     "ex.jpg").astype(np.float32)/255
    # im = histogram_color_transfer(src=im, ref=im2)
    im = lab_color_transfer(src=im, ref=im2)
    im = vibrance(im, 0.5)
    # im = awb(im, useWhite=False)
    im_out = (im*255).astype(np.uint8)
    cv2.imwrite(OUT_PATH / "transfer5.jpg", im_out,
                [int(cv2.IMWRITE_JPEG_QUALITY), 99])

from scipy.ndimage import distance_transform_edt
from pathlib import Path
import cv2
import numpy as np
from scipy.signal import convolve2d

DIR = Path(__file__).resolve().parent
IN_PATH = DIR / "in2"
OUT_PATH = DIR / "out2"


def lopass(im, sigma):
    if (sigma == 0):
        return im
    ksize = 2 * int(3 * sigma) + 1
    f = cv2.GaussianBlur(
        im,
        (ksize, ksize),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT
    )
    return f


def stack(im, mask=False):
    sigmas = np.array([0, 1, 2, 4, 8, 16, 32, 64])
    if (mask):
        sigmas = sigmas[1:]
        sigmas = np.append(sigmas, 128)
        sigmas *= 4

    Gstack = []
    for i, sigma in enumerate(sigmas):
        Gstack.append(lopass(im, sigma))
    Lstack = []
    for i in range(len(Gstack)-1):
        Lstack.append(Gstack[i]-Gstack[i+1])
    Lstack.append(Gstack[-1])
    return np.stack(Gstack), np.stack(Lstack)


def colornorm(src, ref):
    out = src.copy()
    for ch in range(3):
        src_u = np.ma.masked_less(src[:, :, ch], 0.001).mean()
        src_s = np.ma.masked_less(src[:, :, ch], 0.001).std()
        ref_u = np.ma.masked_less(ref[:, :, ch], 0.001).mean()
        ref_s = np.ma.masked_less(ref[:, :, ch], 0.001).std()
        out[:, :, ch] = ((src[:, :, ch] - src_u) / (src_s) * ref_s + ref_u)
    return out


def colornorm2(src, ref):
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    diff = np.ma.masked_less(ref_gray, 0.001).mean(
    ) - np.ma.masked_less(src_gray, 0.001).mean()
    return np.clip(src + diff, 0, 1)


def pad_object(object, mask):
    valid = mask > 0.5
    # pad missing pixels with the nearest colored pixel
    _, indices = distance_transform_edt(~valid, return_indices=True)
    out = object.copy()
    for ch in range(3):
        channel = object[:, :, ch]
        out[:, :, ch] = channel[indices[0], indices[1]]
    out[valid] = out[valid]
    return out


def main(name1, name2, name3):
    l = cv2.imread(IN_PATH / name1).astype(np.float32) / 255.0
    H, W, C = l.shape
    r = cv2.imread(IN_PATH / name2).astype(np.float32) / 255.0
    r = cv2.resize(r, (W, H))
    # r = colornorm(r, l)
    # l = colornorm(l, r)

    mask = np.zeros((H, W), dtype=np.float32)
    mask[:, :W//2] = 1.0
    _, left_lstack = stack(l)
    _, right_lstack = stack(r)

    mask_gstack, _ = stack(mask, True)
    mask_gstack = mask_gstack[..., None]
    lstack = left_lstack * mask_gstack
    rstack = right_lstack * (1-mask_gstack)
    lrstack = lstack + rstack
    answer = np.sum(lrstack, axis=0)
    answer = np.clip(answer, 0, 1)
    cv2.imwrite(OUT_PATH / name3, (answer*255).astype(np.uint8))


# main("orange.png", "apple.png", "oraple.png")


def main2(scene, object, mask, out):
    scene = cv2.imread(scene).astype(np.float32) / 255.0
    H, W, C = scene.shape
    object_rgba = cv2.imread(object, cv2.IMREAD_UNCHANGED).astype(
        np.float32) / 255.0
    object = object_rgba[:, :, :3]

    ### fix colors for better blending ###
    # scene = colornorm(scene, object)
    object = colornorm2(object, scene)

    ### pad empty pixels in object for better blending ###
    fill_mask = object_rgba[:, :, 3]
    object = pad_object(object, fill_mask)

    mask = cv2.imread(mask).astype(np.float32) / 255.0

    # Possible mistake: black outside the extracted object affects filtering?
    # Didn't work
    # object = object * mask + scene * (1 - mask)
    # Better (with alpha)

    print("scene.shape", scene.shape)
    print("objedct.shape", object.shape)
    print("mask.shape", mask.shape)

    _, scene_stack = stack(scene)
    _, object_stack = stack(object)

    mask_stack, _ = stack(mask, True)
    # mask_gstack = mask_gstack[..., None]
    scene = scene_stack * (1-mask_stack)
    object = object_stack * mask_stack
    answer = scene + object
    answer = np.sum(answer, axis=0)
    answer = np.clip(answer, 0, 1)
    cv2.imwrite(out, (answer*255).astype(np.uint8))


# main2(DIR / "work3" / "reyes-campanile-scene.jpg",
#       DIR / "work3" / "reyes-campanile-object.jpg",
#       DIR / "work3" / "reyes-campanile-mask.png",
#       DIR / "work3" / "reyes-campanile.jpg")

# main2(DIR / "work5" / "glacier-volcano-scene.jpg",
#       DIR / "work5" / "glacier-volcano-object.jpg",
#       DIR / "work5" / "glacier-volcano-mask.jpg",
#       DIR / "work5" / "glacier-volcano.jpg")


# main2(DIR / "work6" / "glacier-volcano-scene.jpg",
#       DIR / "work6" / "glacier-volcano-object.jpg",
#       DIR / "work6" / "glacier-volcano-mask.jpg",
#       DIR / "work6" / "glacier-volcano.jpg")

# main2(DIR / "work8" / "glacier-volcano-scene.jpg",
#       DIR / "work8" / "glacier-volcano-object.jpg",
#       DIR / "work8" / "glacier-volcano-mask.png",
#       DIR / "work8" / "glacier-volcano.jpg")

# main2(DIR / "utah-milky" / "utah-milky-scene.jpg",
#       DIR / "utah-milky" / "utah-milky-object.png",  # alpha channel
#       DIR / "utah-milky" / "utah-milky-mask.png",  # lossless
#       DIR / "utah-milky" / "utah-milky.jpg")

# main2(DIR / "reflection" / "reflection1-reflection3-scene.jpg",
#       DIR / "reflection" / "reflection1-reflection3-object.png",
#       DIR / "reflection" / "reflection1-reflection3-mask.png",
#       DIR / "reflection" / "reflection1-reflection3.jpg")

# main2(DIR / "sand-waves" / "sand-waves-scene.jpg",
#       DIR / "sand-waves" / "sand-waves-object.png",
#       DIR / "sand-waves" / "sand-waves-mask.png",
#       DIR / "sand-waves" / "sand-waves.jpg")


# main2(DIR / "forestfire" / "before-after-scene.jpg",
#       DIR / "forestfire" / "before-after-object.png",
#       DIR / "forestfire" / "before-after-mask.png",
#       DIR / "forestfire" / "before-after.jpg")

# main2(DIR / "lakelava" / "before-after-scene.jpg",
#       DIR / "lakelava" / "before-after-object.png",
#       DIR / "lakelava" / "before-after-mask.png",
#       DIR / "lakelava" / "before-after.jpg")

# main2(DIR / "seasons" / "before-after-scene.jpg",
#       DIR / "seasons" / "before-after-object.png",
#       DIR / "seasons" / "before-after-mask.png",
#       DIR / "seasons" / "before-after.jpg")

main2(DIR / "seasons" / "before-after-winter-scene.jpg",
      DIR / "seasons" / "before-after-winter-object.png",
      DIR / "seasons" / "before-after-winter-mask.png",
      DIR / "seasons" / "before-after-winter.jpg")

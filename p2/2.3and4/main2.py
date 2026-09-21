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
    sigmas = np.array([0, 2, 4, 8, 16, 32, 64])
    if (mask):
        # sigmas = sigmas[1:]
        # sigmas = np.append(sigmas, 128)
        sigmas *= 1
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


def main(name1, name2, name3):
    l = cv2.imread(IN_PATH / name1).astype(np.float32) / 255.0
    H, W, C = l.shape
    r = cv2.imread(IN_PATH / name2).astype(np.float32) / 255.0
    r = cv2.resize(r, (W, H))
    # r = colornorm(r, l)
    l = colornorm(l, r)

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
    object = cv2.imread(object).astype(np.float32) / 255.0

    # scene = colornorm(scene, object)
    object = colornorm2(object, scene)

    mask = cv2.imread(mask).astype(np.float32) / 255.0

    # Possible mistake: black outside the extracted object affects filtering?
    # object = object * mask + scene * (1 - mask)

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


# main2(DIR / "work3" / "reyes-campanile-scene.png",
#       DIR / "work3" / "reyes-campanile-object.png",
#       DIR / "work3" / "reyes-campanile-mask.png",
#       DIR / "work3" / "reyes-campanile.png")

# main2(DIR / "work4" / "reflection1-reflection3-scene.jpg",
#       DIR / "work4" / "reflection1-reflection3-object.jpg",
#       DIR / "work4" / "reflection1-reflection3-mask.jpg",
#       DIR / "work4" / "reflection1-reflection3.jpg")

# main2(DIR / "work5" / "glacier-volcano-scene.jpg",
#       DIR / "work5" / "glacier-volcano-object.jpg",
#       DIR / "work5" / "glacier-volcano-mask.jpg",
#       DIR / "work5" / "glacier-volcano.jpg")


# main2(DIR / "work6" / "glacier-volcano-scene.jpg",
#       DIR / "work6" / "glacier-volcano-object.jpg",
#       DIR / "work6" / "glacier-volcano-mask.jpg",
#       DIR / "work6" / "glacier-volcano.jpg")

main2(DIR / "work7" / "utah-milky-scene.jpg",
      DIR / "work7" / "utah-milky-object.jpg",
      DIR / "work7" / "utah-milky-mask.png",  # lossless
      DIR / "work7" / "utah-milky.jpg")

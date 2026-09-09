import numpy as np
import cv2
from pathlib import Path

window_sz = 16


def ncc(u):
    u = u - np.mean(u)
    return (u/np.linalg.norm(u)).flatten()


def align1(u, v):
    top_score = np.inf
    new_d = None
    for dy in range(-window_sz, window_sz+1):
        for dx in range(-window_sz, window_sz+1):
            up = np.roll(u, shift=(dy, dx), axis=(0, 1))
            score = np.dot(ncc(up), ncc(v))
            if score > top_score:
                top_score = score
                new_d = (dy, dx)
    return u


align = align1


DIR = Path(__file__).resolve().parent
IN_PATH = DIR / "in" / "cathedral.jpg"
OUT_PATH = DIR / "out" / "cathedral.jpg"


im = cv2.imread(IN_PATH, cv2.IMREAD_GRAYSCALE)
im = im.astype(np.float32)/255
height = np.floor(im.shape[0] / 3.0).astype(int)


def pad(im):
    p = im.shape[0] // 10
    return cv2.copyMakeBorder(
        im,
        p, p,
        p, p,
        cv2.BORDER_CONSTANT,
        value=0
    )


b = pad(im[:height])
g = pad(im[height: 2*height])
r = pad(im[2*height: 3*height])
print(b.shape)


# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)
ag = align(g, b)
ar = align(r, b)
# create a color image

im_out = np.dstack([ar, ag, b])
im_out = (im_out * 255).astype(int)

# save the image
cv2.imwrite(OUT_PATH, im_out)
# display the image
# cv2.imshow(im_out)
# cv2.show()

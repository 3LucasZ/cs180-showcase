import numpy as np
import cv2
from pathlib import Path

DIR = Path(__file__).resolve().parent
IN_PATH = DIR / "in" / "cathedral.jpg"
OUT_PATH = DIR / "out" / "cathedral.jpg"


im = cv2.imread(IN_PATH, cv2.IMREAD_GRAYSCALE)
im = im.astype(np.float32)/255
height = np.floor(im.shape[0] / 3.0).astype(np.uint)


b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]


def norm(u):
    u = u.flatten()
    u = u - np.mean(u)
    return u/np.linalg.norm(u)


def norm2(u):
    shape = u.shape
    return norm(u).reshape(shape)


b = norm2(b)
g = norm2(g)
r = norm2(r)

im_out = np.hstack([b, g, r])
im_out = ((im_out + 1) * 128).astype(np.uint8)

# save the image
cv2.imwrite(OUT_PATH, im_out)
# display the image
cv2.imshow('Window', im_out)
cv2.waitKey(0)
cv2.destroyAllWindows()

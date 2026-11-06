from pathlib import Path

import cv2
import numpy as np

from B5 import cylindrical_warp
from B4 import auto_mosaic2


DIR = Path(__file__).resolve().parent
IN_PATH = DIR / f"panorama"
OUT_PATH = DIR / f"panorama"
N = 10

ims = [cv2.imread(IN_PATH / f"{i+1}.JPG") for i in range(N)]
f = 0.25
ims = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
       for im in ims]
focal = 1514 * f
ims = [cylindrical_warp(im, focal) for im in ims]

panorama = ims[0]
for i, im in enumerate(ims[1:]):
    print("---STITCH:", i)
    panorama = auto_mosaic2(im, panorama)
cv2.imwrite(OUT_PATH / "panorama.jpg", panorama)

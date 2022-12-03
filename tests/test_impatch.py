import os

import cv2 as opencv
import numpy as np
import matplotlib.pylab as plt

from procs.impatch import impatchify
from utils.plots import imshow


DATA_DIR = 'datasets/DRIVE/training/'
IMG_NAME = '21_training'
PATCH_SIZE = 128

tst_file_img = os.path.join(DATA_DIR, 'images/{}.tif'.format(IMG_NAME))
tst_img = opencv.imread(tst_file_img, opencv.IMREAD_COLOR)

img_patches = impatchify.getPatches(
        imgs=[tst_img],
        patch_size=PATCH_SIZE,
        overlap_ratio=0.
)

# merge patches into image
height, width = tst_img.shape[0:2]
img_merged = np.zeros(shape=tst_img.shape, dtype=tst_img.dtype)

c, r = 1, 1
for patch in img_patches:
    r_start, r_end = PATCH_SIZE * (r - 1), PATCH_SIZE * r
    c_start, c_end = PATCH_SIZE * (c - 1), PATCH_SIZE * c

    patch_rs = r_end - height if r_end > height else 0

    if c_end < width:
        img_merged[r_start:r_end, c_start:c_end, :] = patch[patch_rs:]

        c += 1
    else:
        patch_cs = c_end - width
        img_merged[r_start:r_end, c_start:c_end, :] = patch[patch_rs:, patch_cs:]

        r += 1; c = 1

imshow(img_merged)
plt.show()

assert np.max(img_merged - tst_img) == 0, f'Error patchify image'

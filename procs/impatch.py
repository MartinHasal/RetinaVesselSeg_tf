import numpy as np


class impatchify(object):

    @staticmethod
    def __pathifyOverlap(overlap_ratio: float = .1) -> object:

        # inner implementation of function related to patchify input image with defined
        # overlap ratio
        def impl(img: np.ndarray, patch_size: int) -> list:

            if img.shape[0] < patch_size or img.shape[1] < patch_size:
                dim = img.shape[0:2]
                raise ValueError('Patch size ({}) must be lesser than image dimension {}'.format(patch_size, dim))

            img_patches = []

            rows = img.shape[0]; cols = img.shape[1]
            overlap = int(overlap_ratio * patch_size)

            y_min = y_max = 0
            while y_max < rows:
                y_max = y_min + patch_size
                x_min = x_max = 0

                # check whether last row index (y_max) does not exceed image height
                # if it does, move slice so that y_max equals image height and
                # slice number of rows/cols matches patch_size
                if y_max > rows:
                    y_min -= y_max - rows
                    y_max = rows

                while x_max < cols:
                    x_max = x_min + patch_size

                    # this is same as for y_min/y_max variables above
                    if x_max > cols:
                        x_min -= x_max - cols
                        x_max = cols

                    img_patch = img[y_min:y_max, x_min:x_max]
                    img_patches.append(img_patch)

                    x_min = x_max - overlap

                y_min = y_max - overlap

            return img_patches

        return impl

    @staticmethod
    def __patchifyImg(img: np.ndarray, patch_size: int) -> list:

        if img.shape[0] < patch_size or img.shape[1] < patch_size:
            dim = img.shape[0:2]
            raise ValueError('Patch size ({}) must be lesser than image dimension {}'.format(patch_size, dim))

        img_patches = []
        rows, cols = img.shape[0:2]

        npatch_row = (rows // patch_size) + 1 if (rows % patch_size) != 0 else 0
        npatch_col = (cols // patch_size) + 1 if (cols % patch_size) != 0 else 0

        for y in range(npatch_row):
            for x in range(npatch_col):

                if (x + 1) * patch_size < cols:
                    c_start, c_end = x * patch_size, (x + 1) * patch_size
                else:
                    c_start, c_end = x * patch_size - ((x + 1) * patch_size - cols), cols

                if (y + 1) * patch_size < rows:
                    r_start, r_end = y * patch_size, (y + 1) * patch_size
                else:
                    r_start, r_end = y * patch_size - ((y + 1) * patch_size - rows), rows

                img_patch = img[r_start:r_end, c_start:c_end]
                img_patches.append(img_patch)

        return img_patches

    @staticmethod
    def getPatches(imgs: list, patch_size: int, overlap_ratio: float = 0.) -> list:

        if patch_size < 1:
            raise ValueError('Patch size must be positive integer!'.format(patch_size))

        if overlap_ratio < 0.:
            raise ValueError('Overlap ratio must be non-negative float!')

        # set function that performs splitting image into patches
        fn_patch = impatchify.__pathifyOverlap(overlap_ratio) if overlap_ratio > 0. else impatchify.__patchifyImg

        img_patches = []

        for img in imgs:
            lst_patches = fn_patch(img, patch_size)
            img_patches.extend(lst_patches)

        return img_patches


# tests
if __name__ == '__main__':

    import os

    import matplotlib.pylab as plt
    import cv2 as opencv

    from PIL import Image
    from utils.plots import imshow as imshow, maskshow
    


    DATA_DIR = 'datasets/DRIVE/training/'
    IMG_NAME = '21_training'

    PATCH_SIZE = 128
    OVERLAP_RATIO = 0.1

    tst_file_img = os.path.join(DATA_DIR, 'images/{}.tif'.format(IMG_NAME))
    tst_img = opencv.imread(tst_file_img, opencv.IMREAD_COLOR)

    tst_file_labels = os.path.join(DATA_DIR, '1st_manual/21_manual1.gif')
    tst_labels = Image.open(tst_file_labels)
    tst_labels = np.array(tst_labels)

    tst_file_mask = os.path.join(DATA_DIR, 'mask/{}_mask.gif'.format(IMG_NAME))
    tst_mask = Image.open(tst_file_mask)
    tst_mask = np.array(tst_mask)

    # plot images
    imshow(tst_img, title='Source ({})'.format(IMG_NAME)); plt.show()
    maskshow(tst_labels, title='Labels ({})'.format(IMG_NAME)); plt.show()
    maskshow(tst_mask, title='Mask ({})'.format(IMG_NAME)); plt.show()

    tst_img_patches = impatchify.getPatches(
        imgs=[tst_img],
        patch_size=PATCH_SIZE,
        overlap_ratio=OVERLAP_RATIO
    )

    tst_label_patches = impatchify.getPatches(
        imgs=[tst_labels],
        patch_size=PATCH_SIZE,
        overlap_ratio=OVERLAP_RATIO
    )

    tst_mask_patches = impatchify.getPatches(
        imgs=[tst_mask],
        patch_size=PATCH_SIZE,
        overlap_ratio=OVERLAP_RATIO
    )

    print('#patches_imgs={}'.format(len(tst_img_patches)))
    print('#patches_labels={}'.format(len(tst_label_patches)))
    print('#patches_masks={}'.format(len(tst_mask_patches)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(0, 3):
        imshow(tst_img_patches[i], ax=axes[i], title='patch #{}'.format(i))
    fig.suptitle('Patches with {} overlap ratio (sources)'.format(OVERLAP_RATIO), fontsize=21)
    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(0, 3):
        maskshow(tst_label_patches[i], ax=axes[i], title='patch #{}'.format(i))
    fig.suptitle('Patches with {} overlap ratio (labels)'.format(OVERLAP_RATIO), fontsize=21)
    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(0, 3):
        maskshow(tst_mask_patches[i], ax=axes[i], title='patch #{}'.format(i))
    fig.suptitle('Patches with {} overlap ratio (masks)'.format(OVERLAP_RATIO), fontsize=21)
    fig.tight_layout()
    plt.show()

import gc

import cv2 as opencv
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path as OSPath
from PIL import Image
from sklearn.model_selection import train_test_split

from procs.impatch import impatchify
from utils.timer import elapsed_timer


SHUFFLE_BUFFER_SIZE = 1000


def read_mask(filename):

    # gif cannot be read by opencv
    if filename.split('.')[-1] == 'gif':
        pil_img = Image.open(filename)
        img = np.array(pil_img)
    else:
        img = opencv.imread(filename, opencv.IMREAD_GRAYSCALE)

    img = img / 255
    img = img.astype(np.uint8)

    return img


def aug_left_right(img, mask):

    return tf.cond(pred=tf.random.uniform(()) > 0.5,
                   true_fn=(lambda: (tf.image.flip_left_right(img), tf.image.flip_left_right(mask))),
                   false_fn=(lambda: (img, mask))
                   )


def aug_up_down(img, mask):

    return tf.cond(pred=tf.random.uniform(()) > 0.5,
                   true_fn=(lambda: (tf.image.flip_up_down(img), tf.image.flip_up_down(mask))),
                   false_fn=(lambda: (img, mask))
                   )


def aug_random_brightness(img, mask):

    return tf.image.random_brightness(img, 0.6), mask


class DataAdapter(object):

    def __init__(self, fn_csv: str, patch_size: int = 128, patch_overlap_ratio: int = 0.0, test_ratio: float = 0.2, augmented_ratio: float = .3):

        self._df_paths = None

        self._ds_training = None
        self._ds_test = None

        self._patch_size = -1
        self.patch_size = patch_size

        self._test_ratio = -1
        self.test_ratio = test_ratio

        self._patch_overlap_ratio = 0.
        self.patch_overlap_ratio = patch_overlap_ratio

        self._augmented_ratio = 0.
        self.augmented_ratio = augmented_ratio

        self._fn_csv = None
        self.fn_csv = fn_csv

    @property
    def fn_csv(self):

        return self._fn_csv

    @fn_csv.setter
    def fn_csv(self, fn_csv: str):

        csv_path = OSPath(fn_csv)

        # check if file exists
        if not csv_path.exists():
            msg = 'Source file {} does not exist!'
            raise IOError(msg)

        # check if actual csv path is same as a new one
        if self._fn_csv is not None:
            if csv_path == self._fn_csv:
                return

        self.__reset()
        self._fn_csv = csv_path

    @property
    def patch_size(self) -> int:

        return self._patch_size

    @patch_size.setter
    def patch_size(self, v):

        if v % 32. != 0:
            msg = 'Patch size must be multiply of 32!'
            raise ValueError(msg)

        if v == self.patch_size:
            return

        self.__reset()
        self._patch_size = v

    @property
    def test_ratio(self) -> float:

        return self._test_ratio

    @test_ratio.setter
    def test_ratio(self, value: float) -> None:

        if value <= 0. or value > 1:
            msg = 'Value of test data set ratio must be greater than 0. and less than 1!'
            raise ValueError(msg)

        if value == self.test_ratio:
            return

        self.__reset()
        self._test_ratio = value

    @property
    def augmented_ratio(self) -> float:

        return self._augmented_ratio

    @augmented_ratio.setter
    def augmented_ratio(self, value: float) -> None:

        if value <= 0. or value > 1:
            msg = 'Value of test data set ratio must be greater than 0. and less than 1!'
            raise ValueError(msg)

        if value == self.augmented_ratio:
            return

        self.__reset()
        self._augmented_ratio = value

    @property
    def patch_overlap_ratio(self) -> float:

        return self._patch_overlap_ratio

    @patch_overlap_ratio.setter
    def patch_overlap_ratio(self, value: float) -> None:

        if value < 0. or value >= 1:
            msg = 'Value of test data set ratio must be positive and less than 1!'
            raise ValueError(msg)

    def __reset(self) -> None:

        if self._df_paths is not None:
            del self._df_paths

        if self._ds_training is not None:
            del self._ds_training

        if self._ds_test is not None:
            del self._ds_test

    def __readDataFrame(self) -> None:

        if not self.fn_csv.exists():
            raise IOError('File {} does not exist!'.format(self.fn_csv))

        self._df_paths = pd.read_csv(self.fn_csv, index_col=['index'])

    def __loadImages(self) -> (np.ndarray, np.ndarray):

        col_names = ['PATH_TO_ORIGINAL_IMAGE', 'MASK']

        np_imgs = None
        np_masks = None

        for _, row in self._df_paths[col_names].iterrows():

            # patchify source image
            fn_src = row[col_names[0]]

            with elapsed_timer('Patchifying source {}'.format(fn_src)):
                src_img = opencv.imread(row[col_names[0]], opencv.IMREAD_COLOR)
                src_patches = impatchify.getPatches(
                    imgs=[src_img],
                    patch_size=self.patch_size,
                    overlap_ratio=self.patch_overlap_ratio
                )

            np_patches = np.array(src_patches)
            del src_img
            np_imgs = np.vstack([np_imgs, np_patches]) if np_imgs is not None else np_patches

            # patchify segmentation mask
            fn_mask = row[col_names[1]]

            with elapsed_timer('Patchifying mask {}'.format(fn_mask)):
                mask_img = read_mask(row[col_names[1]])
                mask_patches = impatchify.getPatches(
                    imgs=[mask_img],
                    patch_size=self.patch_size,
                    overlap_ratio=self.patch_overlap_ratio
                )

            np_patches = np.array(mask_patches)
            del mask_img
            np_masks = np.vstack([np_masks, np_patches]) if np_masks is not None else np_patches

        # invoke garbage collector
        gc.collect()

        return np_imgs, np_masks

    def __createDataset(self) -> None:

        try:
            self.__readDataFrame()
        except IOError:
            exit(-1)

        try:
            np_imgs, np_masks = self.__loadImages()
        except IOError:
            exit(-1)

        np_masks = np.expand_dims(np_masks, axis=3)

        # split train and test data set
        imgs, imgs_test, masks, masks_test = train_test_split(np_imgs, np_masks, test_size=self.test_ratio, random_state=42, shuffle=True)

        # create training data set
        ds_train = tf.data.Dataset.from_tensor_slices((imgs, masks))
        ds_train = ds_train.cache().shuffle(SHUFFLE_BUFFER_SIZE)

        if self.augmented_ratio > 0.:
            nsamples = int(len(ds_train) * self.augmented_ratio)
            ds_train_aug = ds_train.take(nsamples).cache().map(aug_left_right).map(aug_up_down).map(aug_random_brightness)

            ds_train = ds_train.concatenate(ds_train_aug)

        # create test data set
        ds_test = tf.data.Dataset.from_tensor_slices((imgs_test, masks_test))
        ds_test = ds_test.cache()

        self._ds_training = ds_train
        self._ds_test = ds_test

    def getTrainingDataset(self):

        if self._ds_training is None or self._ds_test is None:
            self.__createDataset()

        return self._ds_training

    def getTestDataset(self):

        if self._ds_training is None or self._ds_test is None:
            self.__createDataset()

        return self._ds_test


# tests
if __name__ == '__main__':

    DATABASE_CSV_NAME = 'dataset.csv'

    PATH_SIZE = 128
    PATH_OVERLAP_RATIO = .1

    AUGMENTED_RATIO = .5
    TEST_RATIO = .2

    da = DataAdapter(fn_csv=DATABASE_CSV_NAME,
                     patch_size=PATH_SIZE,
                     test_ratio=TEST_RATIO,
                     augmented_ratio=AUGMENTED_RATIO
                     )

    with elapsed_timer('Creating datasets'):
        ds_training = da.getTrainingDataset()

    ds_test = da.getTestDataset()

    print('nsamples_training = {}'.format(len(ds_training)))
    print('nsamples_test = {}'.format(len(ds_test)))

import gc

import cv2 as opencv
import numpy as np
import pandas as pd
import tensorflow as tf

from enum import Enum
from pathlib import Path as OSPath
from typing import Union

from PIL import Image
from sklearn.model_selection import train_test_split

from procs.impatch import impatchify
from utils.timer import elapsed_timer


SHUFFLE_BUFFER_SIZE = 1000


"""
 image/data set augmentation implementation
"""


class DatasetAugmentation(Enum):

    NONE = 0
    CLAHE_EQUALIZATION = 2
    CLAHE_EQUALIZATION_INPLACE = 3
    FLIP_HORIZONTAL = 4
    FLIP_VERTICAL = 8
    ROTATION_90 = 16
    ADJUST_CONTRAST = 32
    ADJUST_BRIGHTNESS = 64
    ALL = 126
    ALL_CLAHE_INPLACE = 127


def getImageEqualization_CLAHE(img, clip_limit=2., tile_grid_size=(8, 8)):
        
    # to HSV format
    hsv_img = opencv.cvtColor(img, opencv.COLOR_BGR2HSV)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

    # create a CLAHE object
    # clip_limit = threshold for contrast limiting.
    # apply CLAHE on V channel
    clahe = opencv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    v = clahe.apply(v)

    # complete image
    hsv_img = np.dstack((h, s, v))

    # convert HSV back to BGR
    img_bgr = opencv.cvtColor(hsv_img, opencv.COLOR_HSV2BGR)

    # invoke garbage collector
    del hsv_img, clahe
    gc.collect()

    return img_bgr


def getPatchRandomFlip_HORIZONTAL(img, mask):

    return tf.cond(pred=tf.random.uniform(()) > 0.5,
                   true_fn=(lambda: (tf.image.flip_left_right(img), tf.image.flip_left_right(mask))),
                   false_fn=(lambda: (img, mask)))


def getPatchRandomFlip_VERTICAL(img, mask):

    return tf.cond(pred=tf.random.uniform(()) > 0.5,
                   true_fn=(lambda: (tf.image.flip_up_down(img), tf.image.flip_up_down(mask))),
                   false_fn=(lambda: (img, mask)))


def getPatchRandomRotate_90(img, mask):

    return tf.cond(pred=tf.random.uniform(()) > 0.5,
                   true_fn=(lambda: (tf.image.rot90(img), tf.image.rot90(mask))),
                   false_fn=(lambda: (img, mask)))


def getPatchRandomBrightness(img, mask):

    return tf.image.random_brightness(img, 0.6), mask


def getPatchRandomAdjust_CONTRAST(img, mask, contrast_range=(0.5, 1.4)):

    # adjust contrast
    contrast = tf.random.uniform(shape=[], minval=contrast_range[0], maxval=contrast_range[1])
    img = tf.image.adjust_contrast(img, contrast)    

    return tf.clip_by_value(img, 0, 1), mask


def getPatchRandomAdjust_BRIGHTNESS(img, mask, brightness_delta=(-0.1, 0.3)):

    # adjust brightness
    brightness = tf.random.uniform(shape=[], minval=brightness_delta[0], maxval=brightness_delta[1])
    img = tf.image.adjust_brightness(img, brightness)

    return tf.clip_by_value(img, 0, 1), mask


"""
 Data set builder implementation
"""


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


class DataAdapter(object):

    def __init__(self,
                 fn_csv: str,
                 patch_size: int = 128,
                 patch_overlap_ratio: float = .0,
                 test_ratio: float = .2,
                 augmentation_ratio: float = .3,
                 augmentation_ops_list: Union[tuple[DatasetAugmentation], list[DatasetAugmentation]] = (DatasetAugmentation.ALL,),
                 augmentation_clahe_ratio: float = .0,
                 contrast_range: Union[tuple[float, float], list[float, float]] = (0.5, 1.4),
                 brightness_delta: Union[tuple[float, float], list[float, float]] = (-0.1, 0.3)):

        self._df_paths = None

        self._ds_training = None
        self._ds_test = None

        self._patch_size = -1
        self.patch_size = patch_size

        self._test_ratio = -1
        self.test_ratio = test_ratio

        self._patch_overlap_ratio = 0.
        self.patch_overlap_ratio = patch_overlap_ratio

        self._augmentation_ratio = 0.
        self.augmentation_ratio = augmentation_ratio

        # augmentation ops
        self._augmentation_ops_list = (DatasetAugmentation.NONE,)
        self._augmentation_ops_flg = DatasetAugmentation.NONE.value
        self.augmentation_ops_list = augmentation_ops_list

        self._augmentation_clahe_ratio = 0.
        self.augmentation_clahe_ratio = augmentation_clahe_ratio

        self._contrast_range = (0., 0.)
        self.contrast_range = contrast_range

        self._brightness_delta = (0., 0.)
        self.brightness_delta = brightness_delta

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
    def augmentation_ratio(self) -> float:

        return self._augmentation_ratio

    @augmentation_ratio.setter
    def augmentation_ratio(self, value: float) -> None:

        if value < 0. or value > 1:
            msg = 'Value of test data set ratio must be greater than 0. and less than 1!'
            raise ValueError(msg)

        if value == self.augmentation_ratio:
            return

        self.__reset()
        self._augmentation_ratio = value

    @property
    def patch_overlap_ratio(self) -> float:

        return self._patch_overlap_ratio

    @patch_overlap_ratio.setter
    def patch_overlap_ratio(self, value: float) -> None:

        if value < 0. or value >= 1:
            msg = 'Value of test data set ratio must be positive and less than 1!'
            raise ValueError(msg)

        self.__reset()
        self._patch_overlap_ratio = value

    @property
    def augmentation_clahe_ratio(self) -> float:

        return self._augmentation_clahe_ratio

    @augmentation_clahe_ratio.setter
    def augmentation_clahe_ratio(self, value: float):

        if value < 0. or value >= 1:
            msg = 'Value of test data set ratio must be positive and less than 1!'
            raise ValueError(msg)

        self.__reset()
        self._augmentation_clahe_ratio = value

    @property
    def augmentation_ops_list(self) -> tuple[DatasetAugmentation]:

        return self._augmentation_ops_list

    @augmentation_ops_list.setter
    def augmentation_ops_list(self, lst_ops: tuple[DatasetAugmentation]) -> None:

        if self._augmentation_ops_list == lst_ops:
            return

        self.__reset()

        flg = 0
        for op in lst_ops: flg |= op.value

        self._augmentation_ops_flg = flg
        self._augmentation_ops_list = lst_ops

    @property
    def contrast_range(self) -> tuple[float, float]:

        return self._contrast_range

    @contrast_range.setter
    def contrast_range(self, rng: tuple[float]) -> None:

        if self._contrast_range == rng:
            return

        self.__reset()
        self._contrast_range = rng

    @property
    def brightness_delta(self) -> tuple[float, float]:

        return self._brightness_delta

    @brightness_delta.setter
    def brightness_delta(self, rng: tuple[float]):

        if self._brightness_delta == rng:
            return

        self.__reset()
        self._brightness_delta = rng

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

    def __loadImages(self, df_img_paths, clahe_augmentation: bool = False) -> (np.ndarray, np.ndarray):

        col_names = ['PATH_TO_ORIGINAL_IMAGE', 'MASK']

        np_imgs = None
        np_masks = None

        for _, row in df_img_paths[col_names].iterrows():

            # patchify source image
            fn_src = row[col_names[0]]

            with elapsed_timer('Patchifying source {}'.format(fn_src)):

                src_img = opencv.imread(row[col_names[0]], opencv.IMREAD_COLOR)

                if clahe_augmentation:
                    src_img = getImageEqualization_CLAHE(src_img)

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
            clahe_augmentation = True if self._augmentation_ops_flg & DatasetAugmentation.CLAHE_EQUALIZATION_INPLACE.value else False
            np_imgs, np_masks = self.__loadImages(self._df_paths, clahe_augmentation=clahe_augmentation)
        except IOError:
            exit(-1)

        if self._augmentation_ops_flg & DatasetAugmentation.CLAHE_EQUALIZATION.value:

            nsamples = int(self.augmentation_clahe_ratio * len(self._df_paths))

            with elapsed_timer('Clahe augmentation (#nsamples={} from training data set)'.format(nsamples)):
                df_paths_random = self._df_paths.sample(nsamples)

                try:
                    np_imgs_clahe, np_masks_clahe = self.__loadImages(df_paths_random, clahe_augmentation=True)
                except IOError:
                    exit(-1)

                np_imgs = np.concatenate((np_imgs, np_imgs_clahe))
                np_masks = np.concatenate((np_masks, np_masks_clahe))

                del np_imgs_clahe, np_masks_clahe
                gc.collect()

        # expand mask dimension
        np_masks = np.expand_dims(np_masks, axis=3)

        # split train and test data set
        imgs, imgs_test, masks, masks_test = train_test_split(np_imgs, np_masks, test_size=self.test_ratio, random_state=42, shuffle=True)

        # create training data set
        ds_train = tf.data.Dataset.from_tensor_slices((imgs, masks))
        ds_train = ds_train.cache()

        if self.augmentation_ratio > 0.:

            nsamples = int(len(ds_train) * self.augmentation_ratio)

            ds_train = ds_train.shuffle(buffer_size=len(ds_train))
            ds_train_aug = (ds_train.take(nsamples).cache())

            # data set augmentation
            if self._augmentation_ops_flg != DatasetAugmentation.NONE.value:

                if self._augmentation_ops_flg & DatasetAugmentation.FLIP_HORIZONTAL.value:
                    with elapsed_timer('Training data set augmentation (random horizontal flip)'):
                        ds_train_aug = ds_train_aug.map(getPatchRandomFlip_HORIZONTAL, num_parallel_calls=tf.data.AUTOTUNE)

                if self._augmentation_ops_flg & DatasetAugmentation.FLIP_VERTICAL.value:
                    with elapsed_timer('Training data set augmentation (random vertical flip)'):
                        ds_train_aug = ds_train_aug.map(getPatchRandomFlip_VERTICAL, num_parallel_calls=tf.data.AUTOTUNE)

                if self._augmentation_ops_flg & DatasetAugmentation.ROTATION_90.value:
                    with elapsed_timer('Training data set augmentation (random rotation 90)'):
                        ds_train_aug = ds_train_aug.map(getPatchRandomRotate_90, num_parallel_calls=tf.data.AUTOTUNE)

                if self._augmentation_ops_flg & DatasetAugmentation.ADJUST_CONTRAST.value:
                    with elapsed_timer('Training data set augmentation (random adjustment contrast)'):
                        ds_train_aug = ds_train_aug.map(getPatchRandomAdjust_CONTRAST, num_parallel_calls=tf.data.AUTOTUNE)

                if self._augmentation_ops_flg & DatasetAugmentation.ADJUST_BRIGHTNESS.value:
                    with elapsed_timer('Training data set augmentation (random adjustment brightness)'):
                        ds_train_aug = ds_train_aug.map(getPatchRandomAdjust_BRIGHTNESS, num_parallel_calls=tf.data.AUTOTUNE)

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


def getDatasets(db_name: str, patch_size: int = 128, patch_overlap_ratio: float = .0, ds_test_ratio: float = .2,
                ds_augmentation_ratio: float = 0., ds_augmentation_ratio_clahe: float = 0.,
                ds_augmentation_ops: list = (DatasetAugmentation.NONE,)) -> (np.ndarray, np.ndarray):

    data_builder = DataAdapter(
        fn_csv=db_name,
        patch_size=patch_size,
        patch_overlap_ratio=patch_overlap_ratio,
        test_ratio=ds_test_ratio,
        augmentation_ratio=ds_augmentation_ratio,
        augmentation_clahe_ratio=ds_augmentation_ratio_clahe,
        augmentation_ops_list=ds_augmentation_ops
    )

    ds_train = data_builder.getTrainingDataset()
    ds_test = data_builder.getTestDataset()

    return ds_train, ds_test


# tests
if __name__ == '__main__':

    import matplotlib.pylab as plt
    from utils.plots import imshow as imshow, maskshow

    DATABASE_CSV_NAME = 'dataset.csv'

    PATH_SIZE = 64
    PATH_OVERLAP_RATIO = .0

    AUGMENTATION_RATIO = .0
    TEST_RATIO = .1

    da = DataAdapter(fn_csv=DATABASE_CSV_NAME,
                     patch_size=PATH_SIZE,
                     test_ratio=TEST_RATIO,
                     augmentation_ratio=AUGMENTATION_RATIO)

    with elapsed_timer('Creating datasets'):
        ds_training = da.getTrainingDataset()

    ds_test = da.getTestDataset()

    print('nsamples_training = {}'.format(len(ds_training)))
    print('nsamples_test = {}'.format(len(ds_test)))

    # plot some predictions
    nsamples = 4
    fig, axes = plt.subplots(nsamples, 2, figsize=(8, 8))
    for ds_sample, i in zip(ds_test.take(nsamples), range(nsamples)):
        imshow(ds_sample[0].numpy(), ax=axes[i][0])

        maskshow(ds_sample[1].numpy(), ax=axes[i][1])
    fig.tight_layout()
    plt.show()

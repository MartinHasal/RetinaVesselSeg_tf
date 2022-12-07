import argparse
import os

import cv2 as opencv
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from PIL import Image

from imblearn.metrics import classification_report_imbalanced

from funcs.inference import predict, predictImg
from funcs.train import trainSegmentationModel
from models.unet import UNet
from procs.adapter import DataAdapter
from utils.cmat import ConfusionMatrix
from utils.plots import imshow, maskshow, plotTrainingHistory
from utils.timer import elapsed_timer


def getDatasets(db_name: str, patch_size: int = 128, overlap_ratio: float = .0, ds_test_ratio: float = .2,
                augmented_ratio: float = 0., clahe_enhancement: bool = False) -> (np.ndarray, np.ndarray):

    da = DataAdapter(fn_csv=db_name,
                     patch_size=patch_size,
                     test_ratio=ds_test_ratio,
                     patch_overlap_ratio=overlap_ratio,
                     augmented_ratio=augmented_ratio,
                     enhancement=clahe_enhancement
                     )

    ds_train = da.getTrainingDataset()
    ds_test = da.getTestDataset()

    return ds_train, ds_test


def buildModel(input_shape, nclasses: int = 2, encoder_type: str = 'vgg16'):

    unet = UNet(input_shape, nclasses=nclasses, encoder_type=encoder_type)
    nn_unet_vgg16 = unet.model
    nn_unet_vgg16.summary()

    return nn_unet_vgg16


def predictTestDataset(ds, nsamples_to_plot: int, nn_model) -> None:

    y_prob, y_label = predict(nn_model, ds)

    fig, axes = plt.subplots(nsamples_to_plot, 4, figsize=(8, 8))
    for idx, ds_sample in enumerate(ds_test.take(nsamples_to_plot)):
        imshow(ds_sample[0].numpy(), ax=axes[idx][0], title='Input image')
        maskshow(ds_sample[1].numpy(), ax=axes[idx][1], title='Mask (true)')
        maskshow(y_prob[idx], ax=axes[idx][2], title='Mask (pred. prob.f)')
        maskshow(y_label[idx], ax=axes[idx][3], title='Mask (pred. label)')
    fig.suptitle('Predictions on test data set')
    fig.tight_layout()
    plt.show()

    # get ground true
    y_true = np.concatenate([y for _, y in ds], axis=0).reshape(-1)

    # plot confusion matrix
    class_names = ['Background', 'Vessel']
    cm = ConfusionMatrix(ds, y_label.numpy(), class_names)
    cm.plot(figsize=(4, 4), title_fontsize=14, label_fontsize=12, ticks_fontsize=10, value_size=8)

    # print classification reports
    print(classification_report_imbalanced(y_true.reshape(-1), y_label.numpy().reshape(-1)))


def plotPredictedImg(fn_img: str, fn_label: str, nn_model) -> None:

    img = opencv.imread(fn_img, opencv.IMREAD_COLOR)

    pil_img = Image.open(fn_label)
    label = np.array(pil_img) / 255; label = label.astype(np.uint8)

    predicted_prob, predicted_label = predictImg(nn_model, img)

    mask_titles = ['Mask (true)', 'Mask (pred. probability)', 'Mask (pred. label)']
    mask_imgs = [label, predicted_prob, predicted_label]

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    # plot source image
    imshow(img, ax=axes[0], title='Input image')
    # plot mask
    for mask, title, idx in zip(mask_imgs, mask_titles, range(1, 4)):
        maskshow(mask, ax=axes[idx], title=title)
    fig.suptitle('Image {}'.format(fn_img))
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--db_csv',
                        metavar='CSV_FILE',
                        type=str,
                        default='dataset.csv',
                        required=False)

    parser.add_argument('--output_model_path',
                        metavar='PATH',
                        required=False)

    parser.add_argument('--output_model_name',
                        metavar='MODEL_NAME',
                        default='unet_vgg16',
                        required=False)

    args = parser.parse_args()

    # data builder settings
    DATABASE_CSV_NAME = args.db_csv

    PATCH_SIZE = 128
    PATCH_OVERLAP_RATIO = .2

    AUGMENTED_RATIO = .5
    DS_TEST_RATIO = .2

    CLAHE_ENHANCEMENT = False

    # NN settings
    IMG_SHAPE = (PATCH_SIZE, PATCH_SIZE, 3)
    NCLASSES = 2

    BATCH_SIZE = 16
    NEPOCHS = 30

    # pipeline running

    with elapsed_timer('Creating datasets'):

        ds_train, ds_test = getDatasets(DATABASE_CSV_NAME,
                                        patch_size=PATCH_SIZE,
                                        overlap_ratio=PATCH_OVERLAP_RATIO,
                                        ds_test_ratio=DS_TEST_RATIO,
                                        clahe_enhancement=CLAHE_ENHANCEMENT)

    with elapsed_timer('Build models'):

        nn_unet_vgg16 = buildModel(IMG_SHAPE, NCLASSES)

    with elapsed_timer('Training model'):

        history = trainSegmentationModel(nn_model=nn_unet_vgg16,
                                         nclasses=NCLASSES,
                                         ds_train=ds_train,
                                         ds_val=ds_test,
                                         nepochs=NEPOCHS,
                                         batch_size=BATCH_SIZE)

    # plot training history
    df_history = pd.DataFrame(history.history)
    plotTrainingHistory(df_history)

    # predict on image
    DATA_DIR = 'datasets/DRIVE/training/'
    IMG_NAME = '38_training'
    LABEL_NAME = '38_manual1'

    path_tst_img = os.path.join(DATA_DIR, 'images/{}.tif'.format(IMG_NAME))
    path_tst_label = os.path.join(DATA_DIR, '1st_manual/{}.gif'.format(LABEL_NAME))

    # predict on test data set
    NSAMPLES = 4
    predictTestDataset(ds_test, nsamples_to_plot=NSAMPLES, nn_model=nn_unet_vgg16)

    # plot predicting images
    plotPredictedImg(path_tst_img, path_tst_label, nn_model=nn_unet_vgg16)

    # saving models
    if args.output_model_path is not None:
        with elapsed_timer('Saving model'):
            if not os.path.exists(args.output_model_path):
                os.makedirs(args.output_model_path)
            nn_unet_vgg16.save(os.path.join(args.output_model_path, args.output_model_name))


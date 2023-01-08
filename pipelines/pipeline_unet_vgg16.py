import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from imblearn.metrics import classification_report_imbalanced

from funcs.inference import predict, predictImg
from funcs.losses import LossType
from funcs.train import trainSegmentationModel
from models.unet import UNet
from procs.adapter import DataAdapter
from utils.cmat import ConfusionMatrix
from utils.plots import imshow, maskshow, plotTrainingHistory, plotColorizedVessels, plotPredictedImg
from utils.plots import plotHistogramImgSlicer, plotPredictedImgSlicer
from utils.timer import elapsed_timer
from utils.roc import AucRoc


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


def buildModel(input_shape, nclasses: int = 2, encoder_type: str = 'vgg16', trainable_encoder: bool = False):

    unet = UNet(input_shape, nclasses=nclasses, encoder_type=encoder_type, trainable_encoder=trainable_encoder)
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
    y_true = np.concatenate([y for _, y in ds], axis=0).reshape(-1).astype(np.float32)

    class_names = ['Background', 'Vessel']
    cm = ConfusionMatrix(ds, y_label, class_names)
    cm.plot(figsize=(4, 4), title_fontsize=14, label_fontsize=12, ticks_fontsize=10, value_size=8)

    # print classification report (label)
    print('Classification report (labels)')
    print(classification_report_imbalanced(y_true, y_label.reshape(-1)))

    # plot auc roc curve
    auc_roc = AucRoc(y_true=y_true, y_pred=y_prob)
    auc_roc.plot()
    plt.show()

    #
    print('auc roc = {0:.4f}'.format(auc_roc.auc))


""" dev notes 

ZZZ = 'D:\\Dropbox (ARG@CS.FEI.VSB)\\Dataset - retiny\\images_from_doctor\\ML\\SEGMENTATION\\RetinaVesselSeg_tf\\datasets\\DRIVE\\training\\images\\21_training.tif'
img = opencv.imread(ZZZ, opencv.IMREAD_COLOR)
plt.imshow(img)
predicted_prob, predicted_label = predictImg(nn_unet_vgg16, img)
plt.imshow(predicted_prob,cmap='gray')

"""

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

    parser.add_argument('--patch_size',
                        metavar='PATCH_SIZE',
                        default=128,
                        required=False)

    parser.add_argument('--patch_overlap_ratio',
                        metavar='PATCH_OVERLAP_RATIO',
                        default=.5,
                        required=False)

    parser.add_argument('--ds_augmented_ratio',
                        metavar='AUGMENTED_RATIO',
                        default=1,
                        required=False)

    parser.add_argument('--ds_test_ratio',
                        metavar='TEST_DATASET_RATIO',
                        default=.1,
                        required=False)

    parser.add_argument('--batch_size',
                        metavar='BATCH_SIZE',
                        default=32,
                        required=False)

    parser.add_argument('--nepochs',
                        metavar='NUMBER_OF_EPOCHS',
                        default=30,
                        required=False)

    parser.add_argument('--loss_type',
                        metavar='LOSS FUNCTION TYPE',
                        type=LossType,
                        choices=LossType,
                        default=LossType.CROSS_ENTROPY,
                        required=False)

    args = parser.parse_args()

    # data set builder settings
    DATABASE_CSV_NAME = args.db_csv

    PATCH_SIZE = args.patch_size
    PATCH_OVERLAP_RATIO = args.patch_overlap_ratio

    AUGMENTED_RATIO = args.ds_augmented_ratio
    DS_TEST_RATIO = args.ds_test_ratio

    CLAHE_ENHANCEMENT = False

    # NN settings
    IMG_SHAPE = (PATCH_SIZE, PATCH_SIZE, 3)
    NCLASSES = 2

    BATCH_SIZE = args.batch_size
    NEPOCHS = args.nepochs

    LOSS_TYPE = args.loss_type

    # pipeline running

    with elapsed_timer('Creating datasets'):

        ds_train, ds_test = getDatasets(DATABASE_CSV_NAME,
                                        patch_size=PATCH_SIZE,
                                        overlap_ratio=PATCH_OVERLAP_RATIO,
                                        ds_test_ratio=DS_TEST_RATIO,
                                        clahe_enhancement=CLAHE_ENHANCEMENT)

    with elapsed_timer('Build models'):

        nn_unet_vgg16 = buildModel(IMG_SHAPE, NCLASSES, trainable_encoder=True)

    with elapsed_timer('Training model'):

        history = trainSegmentationModel(nn_model=nn_unet_vgg16,
                                         nclasses=NCLASSES,
                                         ds_train=ds_train,
                                         ds_val=ds_test,
                                         nepochs=NEPOCHS,
                                         batch_size=BATCH_SIZE,
                                         loss_type=LOSS_TYPE,
                                         decay='warmup')

    # plot training history
    df_history = pd.DataFrame(history.history)
    plotTrainingHistory(df_history)

    # predict on image
    DATA_DIR = 'datasets/DRIVE/training/'
    IMG_NAME = '23_training'
    LABEL_NAME = '23_manual1'

    path_tst_img = os.path.join(DATA_DIR, 'images/{}.tif'.format(IMG_NAME))
    path_tst_label = os.path.join(DATA_DIR, '1st_manual/{}.gif'.format(LABEL_NAME))

    # predict on test data set
    NSAMPLES = 4
    predictTestDataset(ds_test, nsamples_to_plot=NSAMPLES, nn_model=nn_unet_vgg16)

    # plot predicting images
    plotPredictedImg(path_tst_img, path_tst_label, predictImg, nn_model=nn_unet_vgg16)
    
    plotPredictedImgSlicer(path_tst_img, path_tst_label, predictImg, nn_model=nn_unet_vgg16)
    plotColorizedVessels(path_tst_img, predictImg, nn_model=nn_unet_vgg16)
    plotHistogramImgSlicer(path_tst_img, path_tst_label, predictImg, nn_model=nn_unet_vgg16)

    # saving model
    if args.output_model_path is not None:
        with elapsed_timer('Saving model'):
            if not os.path.exists(args.output_model_path):
                os.makedirs(args.output_model_path)
            nn_unet_vgg16.save(os.path.join(args.output_model_path, args.output_model_name))

import os

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from imblearn.metrics import classification_report_imbalanced

from pipelines.args import cli_argument_parser
from funcs.inference import predict, predictImg
from funcs.train import trainSegmentationModel
from models.unet import UNet
from procs.adapter import getDatasets
from utils.cmat import ConfusionMatrix
from utils.plots import imshow, maskshow, plotTrainingHistory, plotColorizedVessels, plotPredictedImg
from utils.plots import plotHistogramImgSlicer, plotPredictedImgSlicer
from utils.timer import elapsed_timer
from utils.roc import AucRoc


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

    kwargs = cli_argument_parser()

    # pipeline running

    with elapsed_timer('Creating datasets'):

        ds_train, ds_test = getDatasets(
            db_name=kwargs['db_name'],
            patch_size=kwargs['patch_size'],
            patch_overlap_ratio=kwargs['patch_overlap_ratio'],
            ds_test_ratio=kwargs['ds_test_ratio'],
            ds_augmentation_ratio=kwargs['ds_augmentation_ratio'],
            ds_augmentation_ratio_clahe=kwargs['clahe_augmentation_ratio'],
            ds_augmentation_ops=kwargs['ds_augmentation_ops']
        )

    with elapsed_timer('Build models'):

        IMG_SHAPE = (kwargs['patch_size'], kwargs['patch_size'], 3)
        NCLASSES = 2

        nn_unet_vgg16 = buildModel(IMG_SHAPE, NCLASSES, trainable_encoder=True)

    with elapsed_timer('Training model'):

        history = trainSegmentationModel(nn_model=nn_unet_vgg16,
                                         nclasses=NCLASSES,
                                         ds_train=ds_train,
                                         ds_val=ds_test,
                                         nepochs=kwargs['nepochs'],
                                         batch_size=kwargs['batch_size'],
                                         loss_type=kwargs['loss_type'],
                                         decay=kwargs['lr_decay_type'])

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
    OUTPUT_MODEL_PATH = kwargs['output_model_path']
    OUTPUT_MODEL_NAME = kwargs['output_model_name']

    if OUTPUT_MODEL_PATH is not None:
        with elapsed_timer('Saving model'):

            if not os.path.exists(OUTPUT_MODEL_PATH):
                os.makedirs(OUTPUT_MODEL_PATH)

            fn = os.path.join(OUTPUT_MODEL_PATH, OUTPUT_MODEL_NAME)
            nn_unet_vgg16.save(os.path.join(OUTPUT_MODEL_PATH, OUTPUT_MODEL_NAME))

import os

import cv2 as opencv
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from PIL import Image

from funcs.inference import predict, predictImg
from funcs.train import trainSegmentationModel
from models.unet import UNet
from procs.adapter import DataAdapter
from utils.plots import imshow, maskshow, plotTrainingHistory
from utils.timer import elapsed_timer


DATABASE_CSV_NAME = 'dataset.csv'

PATH_SIZE = 128
PATH_OVERLAP_RATIO = .2

AUGMENTED_RATIO = .5
TEST_RATIO = .2

da = DataAdapter(fn_csv=DATABASE_CSV_NAME,
                 patch_size=PATH_SIZE,
                 test_ratio=TEST_RATIO,
                 augmented_ratio=AUGMENTED_RATIO
                 )

with elapsed_timer('Creating datasets'):
    ds_train = da.getTrainingDataset()
    ds_test = da.getTestDataset()

# building unet with VGG16 encoder
IMG_SHAPE = (PATH_SIZE, PATH_SIZE, 3)
NCLASSES = 2

BATCH_SIZE = 16
NEPOCHS = 30

unet = UNet(IMG_SHAPE, nclasses=NCLASSES, encoder_type='vgg16')
nn_unet_vgg16 = unet.model
nn_unet_vgg16.summary()

# training model using ADAM solver and cross entropy (sparse format) as loss function
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

# prediction on test data set
y_prob, y_label = predict(nn_unet_vgg16, ds_test)

nsamples = 4

fig, axes = plt.subplots(nsamples, 3, figsize=(8, 8))
for ds_sample, i in zip(ds_test.take(nsamples), range(nsamples)):
    imshow(ds_sample[0].numpy(), ax=axes[i][0], title='Input image')

    maskshow(ds_sample[1].numpy(), ax=axes[i][1], title='Mask (true)')
    maskshow(y_label[i], ax=axes[i][2], title='Mask (predicted)')
fig.suptitle('Predictions on test data set')
fig.tight_layout()
plt.show()

# predict on image
DATA_DIR = 'datasets/DRIVE/training/'
IMG_NAME = '21_training'
LABEL_NAME = '21_manual1'

tst_file_img = os.path.join(DATA_DIR, 'images/{}.tif'.format(IMG_NAME))
tst_img = opencv.imread(tst_file_img, opencv.IMREAD_COLOR)

pil_img = Image.open(os.path.join(DATA_DIR, '1st_manual/{}.gif'.format(LABEL_NAME)))
label = np.array(pil_img)
label = label / 255
label = label.astype(np.uint8)

predicted_label = predictImg(nn_unet_vgg16, tst_img)

fig, axes = plt.subplots(1, 3, figsize=(8, 3))
imshow(tst_img, ax=axes[0], title='Input image')
maskshow(label, ax=axes[1], title='Mask (true)')
maskshow(predicted_label, ax=axes[2], title='Mask (predicted)')
fig.suptitle('File {} ({})'.format(IMG_NAME, DATA_DIR))
plt.show()

import numpy as np
import tensorflow as tf

import matplotlib.pylab as plt

from keras.engine.functional import Functional as KerasFunctional
from imblearn.metrics import classification_report_imbalanced
from tensorflow.data import Dataset as TFDataset

from procs.impatch import impatchify
from utils.cmat import ConfusionMatrix
from utils.plots import imshow, maskshow
from utils.roc import AucRoc


def predict(nn_model: KerasFunctional, ds: tf.data.Dataset, batch_size: int = 32) -> [np.ndarray, np.ndarray]:

    ds_batches = (
        ds.cache()
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    y_prob = nn_model.predict(ds_batches)
    y_label = tf.math.argmax(y_prob, axis=-1)

    # convert to P(X | y = 1)
    y_prob = tf.math.reduce_max(y_prob, axis=-1)
    y_prob = np.abs(np.ones(y_label.shape, dtype=np.float32) - tf.cast(y_label, dtype=np.float32) - y_prob)

    return y_prob, y_label.numpy()


def predictImg(nn_model: KerasFunctional, img: np.ndarray, patch_size: int = 128) -> (np.ndarray, np.ndarray):

    img_patches = impatchify.getPatches(
        imgs=[img],
        patch_size=patch_size,
        overlap_ratio=0.
    )

    bs = len(img_patches)
    ds = TFDataset.from_tensor_slices(img_patches).cache().batch(bs).prefetch(buffer_size=tf.data.AUTOTUNE)

    mask_prob = nn_model.predict(ds)
    mask_label = tf.math.argmax(mask_prob, axis=-1)

    # merge patches
    height, width = img.shape[:2]
    img_labels = np.zeros(shape=img.shape[:2], dtype=np.uint8)
    img_prob = np.zeros(shape=img.shape[:2], dtype=np.float)

    c = r = 1
    for patch_prob, patch_label in zip(mask_prob, mask_label):

        r_start, r_end = patch_size * (r - 1), patch_size * r
        c_start, c_end = patch_size * (c - 1), patch_size * c

        patch_rs = r_end - height if r_end > height else 0

        if c_end < width:
            img_labels[r_start:r_end, c_start:c_end] = patch_label[patch_rs:]
            img_prob[r_start:r_end, c_start:c_end] = tf.math.reduce_max(patch_prob[patch_rs:], axis=-1)
            c += 1
        else:
            patch_cs = c_end - width
            img_labels[r_start:r_end, c_start:c_end] = patch_label[patch_rs:, patch_cs:]

            tmp = tf.math.reduce_max(patch_prob[patch_rs:, patch_cs:], axis=-1)
            img_prob[r_start:r_end, c_start:c_end] = tmp
            r += 1; c = 1

    # convert to P(X | y = 1)
    # img_prob = np.abs(np.ones(img_labels.shape, dtype=np.float32) - img_labels - img_prob)

    return img_prob, img_labels


def predictDataset(ds, nsamples_to_plot: int, nn_model) -> None:

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
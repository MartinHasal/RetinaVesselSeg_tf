import tensorflow as tf
import numpy as np

from tensorflow.data import Dataset as TFDataset
from keras.engine.functional import Functional as KerasFunctional
from procs.impatch import impatchify


def predict(nn_model: KerasFunctional, ds: tf.data.Dataset, batch_size: int = 32) -> [np.ndarray, np.ndarray]:

    ds_batches = (
        ds.cache()
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    y_prob = nn_model.predict(ds_batches)
    y_label = tf.math.argmax(y_prob, axis=-1)

    return y_prob, y_label


def predictImg(nn_model: KerasFunctional, img: np.ndarray) -> np.ndarray:

    patch_size = 128

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

    c = r = 1
    for patch_label in mask_label:

        r_start, r_end = patch_size * (r - 1), patch_size * r
        c_start, c_end = patch_size * (c - 1), patch_size * c

        patch_rs = r_end - height if r_end > height else 0

        if c_end < width:
            img_labels[r_start:r_end, c_start:c_end] = patch_label[patch_rs:]
            c += 1
        else:
            patch_cs = c_end - width
            img_labels[r_start:r_end, c_start:c_end] = patch_label[patch_rs:, patch_cs:]

            r += 1; c = 1

    return img_labels

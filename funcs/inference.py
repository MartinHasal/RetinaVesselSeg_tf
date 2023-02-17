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

from utils.smooth_blender_predicitions import predict_img_with_smooth_windowing


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
    for idx, ds_sample in enumerate(ds.take(nsamples_to_plot)):
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
    

""" 
# generally default serving function is prepared for 
# prediction on images, but model is trained to predict single patch.
# Hence default serving function is not optimal solution and
# model predicition must be handled by more complex function, which
# calls input signature, like inside this function.  
@tf.function(input_signature=[tf.TensorSpec([None,], dtype=tf.string),
                              tf.TensorSpec([], dtype=tf.int64)])
def predictMasks(filenames, patch_size):

    #@tf.function
    def predictSingleImage(image):
        return predictImg(nn_model, image, patch_size)
        
    img_prob, img_labels = tf.map_fn(
        predictSingleImage,
        filenames,
        fn_output_signature=(np.ndarray, np.ndarray)
    )
    
    return {
        'image_path': filenames,
        'mask_prob': img_prob,
        'mask_label': img_labels
    }
"""   

def predictListOfFiles(nn_model: KerasFunctional, images_paths: list, patch_size: int = 128, blending=False) -> dict[str, dict[np.ndarray,np.ndarray, np.ndarray]]:
    """ Function computing prediction on list of images 
        It is used to load images of children.
        It does multiple things:
            loads image
            bool: smooth the patches
            applies the treshold
            returns {images original, mask_prob, mask label}
    """
    import cv2 as opencv
    from utils.plots import convertProbability
    
    return_values = {}
    for k, img_path in enumerate(images_paths):
        img = opencv.imread(img_path, opencv.IMREAD_COLOR)
        
        if blending:
            # it produces results for two categories probability to be background 
            # and probability to be segmented object, as in this case there 
            # are only two categories their sum is 1 for every pixel
            # and as probability we can you any of them
            # as important is to segment vessels [:,:,1] is used
            print(f'Processing image {k}/{len(images_paths)}')
            predictions_smooth = predict_img_with_smooth_windowing(
                img,
                window_size = patch_size,
                subdivisions = 2,
                nb_classes = 2,
                pred_func = (
                    #lambda img_bath_subdiv: np.argmax (nn_unet_vgg16.predict(img_bath_subdiv), axis = -1) 
                    lambda img_bath_subdiv: nn_model.predict(img_bath_subdiv)
                    #predictImg(nn_unet_vgg16, img)[1]
                    #lambda img: predict(nn_unet_vgg16, img)[1]
                    )
                )
            predicted_prob = predictions_smooth[:,:,1]
            predicted_label = np.argmax(predictions_smooth, axis = 2)
            predicted_label = predicted_label.astype(np.uint8) # necesarry for opencv
            
        else:
            predicted_prob, predicted_label = predictImg(nn_model, img)
            predicted_prob = convertProbability(predicted_prob, predicted_label)
        
        results = {
                   'image': img,
                   'mask_prob': predicted_prob,
                   'mask_label': predicted_label
                   }
        return_values[img_path] = results
                                 
            
    return return_values
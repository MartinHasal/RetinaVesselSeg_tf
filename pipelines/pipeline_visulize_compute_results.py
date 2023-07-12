import os

import pandas as pd
import numpy as np

from pipelines.args import cli_argument_parser
from funcs.inference import predictDataset, predictImg
from models.unet import UNet
from procs.adapter import getDatasets
from utils.plots import plotColorizedVessels, plotPredictedImg
from utils.plots import (plotTrainingHistory, 
                         plotColorizedVessels, 
                         plotPredictedImg, 
                         clean_image, 
                         plotListofImages, 
                         plotPredictedImg2x2,
                         plotPredictedImgSlicer,
                         plotHistogramImgSlicer)

from utils.timer import elapsed_timer
from utils.model import save_model
import tensorflow as tf
from utils.smooth_blender_predicitions import predict_img_with_smooth_windowing
from cv2 import imread



def buildModel(input_shape, nclasses: int = 2, encoder_type: str = 'vgg16', trainable_encoder: bool = False):

    unet = UNet(input_shape, nclasses=nclasses, encoder_type=encoder_type, trainable_encoder=trainable_encoder)
    nn_unet = unet.model
    nn_unet.summary()

    return nn_unet


""" dev notes 

ZZZ = 'D:\\Dropbox (ARG@CS.FEI.VSB)\\Dataset - retiny\\images_from_doctor\\ML\\SEGMENTATION\\RetinaVesselSeg_tf\\datasets\\DRIVE\\training\\images\\21_training.tif'
img = opencv.imread(ZZZ, opencv.IMREAD_COLOR)
plt.imshow(img)
predicted_prob, predicted_label = predictImg(nn_unet_vgg16, img)
plt.imshow(predicted_prob,cmap='gray')


pipelines/pipeline_unet_vgg16.py --db_csv data_paths.csv --crop_val 21 --patch_overlap_ratio 0.5 --ds_augmentation_ratio 0.8 --lr_decay_type 'warmup' --model_trainable_encoder True --nepochs 30 --output_model_path '/content/RetinaVesselSeg_tf/model'
"""

IMG_SHAPE = (128, 128, 3)
NCLASSES = 2

nn_unet_vgg16 = buildModel(IMG_SHAPE, NCLASSES, trainable_encoder=True)

nn_unet_vgg16.load_weights('.\\model\\unet_vgg16_best\\variables\\variables')

#localhost_load_option = tf.saved_model.LoadOptions(experimental_io_device="/job:localhost")
#nn_unet_vgg16 = tf.saved_model.load('model/unet_vgg16', options=localhost_load_option)



# visualize the results

DATASET_PATH = 'data_paths.csv'

df = pd.read_csv(DATASET_PATH)



# patched data
with elapsed_timer('Creating datasets'):

    ds_train, ds_test = getDatasets(
        db_name=DATASET_PATH,
        patch_size=128,
        patch_overlap_ratio=0.5,
        ds_test_ratio=0.1,
        ds_augmentation_ratio=0.8,
    )

NSAMPLES = 4
predictDataset(ds_test, nsamples_to_plot=NSAMPLES, nn_model=nn_unet_vgg16)

images_Z = list(df['PATH_TO_ORIGINAL_IMAGE'][:3].values)


def blending_cleaning(path: str, cleaning: float = 0, probt: float = 0.5, plot: bool = True) -> None:


    predictions_smooth = predict_img_with_smooth_windowing(
            imread(path),
            window_size = 128,
            subdivisions = 2,
            nb_classes = 2,
            pred_func = (
                #lambda img_bath_subdiv: np.argmax (nn_unet_vgg16.predict(img_bath_subdiv), axis = -1) 
                lambda img_bath_subdiv: nn_unet_vgg16.predict(img_bath_subdiv)
                #predictImg(nn_unet_vgg16, img)[1]
                #lambda img: predict(nn_unet_vgg16, img)[1]
                )
            )

    final_prediction = np.argmax(predictions_smooth, axis = 2)
    final_prediction = (final_prediction*255).astype(np.uint8)
    
    if cleaning:
        cleaned = clean_image((final_prediction), percentage=cleaning,prob_threshold=probt, plot=True)
        plotColorizedVessels(path, predictImg, nn_model=nn_unet_vgg16, blended=cleaned) 
        return cleaned
    else:
        plotColorizedVessels(path, predictImg, nn_model=nn_unet_vgg16, blended=final_prediction) 
        return final_prediction




DATA_DIR = 'datasets/DRIVE/training/'
IMG_NAME = '21_training'
LABEL_NAME = '21_manual1'

path_tst_img = os.path.join(DATA_DIR, 'images/{}.tif'.format(IMG_NAME))
path_tst_label = os.path.join(DATA_DIR, '1st_manual/{}.gif'.format(LABEL_NAME))

plotPredictedImg(path_tst_img, path_tst_label, predictImg, nn_model=nn_unet_vgg16)

plotPredictedImg2x2(path_tst_img, path_tst_label, predictImg, nn_model=nn_unet_vgg16)


child_path = 'D:\\Dropbox (ARG@CS.FEI.VSB)\\Dataset - retiny\\images_from_doctor\\ML\\images_stack\\013_F_GA39_BW3390_PA40_DG111_DG20_PF0_D1_S01_4.jpg'
blending_cleaning(child_path, cleaning=5e-2, probt=0.95)

child_path = 'D:\\Dropbox (ARG@CS.FEI.VSB)\\Dataset - retiny\\images_from_doctor\\ML\\SEGMENTATION\\RetCamImageProcs\\obr_test\\image-002.jpg'


# plot predicting images
plotPredictedImg(path_tst_img, path_tst_label, predictImg, nn_model=nn_unet_vgg16)
plotPredictedImgSlicer(path_tst_img, path_tst_label, predictImg, nn_model=nn_unet_vgg16)

plotColorizedVessels(path_tst_img, predictImg, nn_model=nn_unet_vgg16)
plotHistogramImgSlicer(path_tst_img, path_tst_label, predictImg, nn_model=nn_unet_vgg16)

# Edges 

import cv2
from matplotlib import pyplot as plt
img = cv2.imread(path_tst_img)
img_prob, img_labels = predictImg(nn_unet_vgg16, img)


def visualize_numpy_array(arr):
    fig = plt.figure()
    plt.imshow(arr)
    plt.show()
    
visualize_numpy_array(img_prob)
visualize_numpy_array(img)


predictions_smooth = predict_img_with_smooth_windowing(
        img,
        window_size = 128,
        subdivisions = 2,
        nb_classes = 2,
        pred_func = (
            #lambda img_bath_subdiv: np.argmax (nn_unet_vgg16.predict(img_bath_subdiv), axis = -1) 
            lambda img_bath_subdiv: nn_unet_vgg16.predict(img_bath_subdiv)
            #predictImg(nn_unet_vgg16, img)[1]
            #lambda img: predict(nn_unet_vgg16, img)[1]
            )
        )

visualize_numpy_array(predictions_smooth[:,:,0])
visualize_numpy_array(predictions_smooth[:,:,1])

pokus = np.abs ( predictions_smooth[:,:,0] - predictions_smooth[:,:,1] )
visualize_numpy_array(pokus)

threshold = 0.5
pokus_thr = np.where(pokus > threshold, 0, 1)
visualize_numpy_array(pokus_thr)

plt.imshow(pokus_thr, cmap='gray', vmin=0, vmax=1)


final_prediction = np.argmax(predictions_smooth, axis = 2)
final_prediction = (final_prediction*255).astype(np.uint8)

visualize_numpy_array(final_prediction)

img_small = img[200:328,200:328,:]
img_small = np.expand_dims(img_small, axis=0)
pred_prob = nn_unet_vgg16.predict(img_small)
visualize_numpy_array(pred_prob[0,:,:,0])
visualize_numpy_array(pred_prob[0,:,:,1])


""" 
fig, axs = plt.subplots(1, 2, figsize=(10,5))

# Plot the first image on the left subplot
axs[0].imshow(plt.imread(path_tst_img))
axs[0].set_title('Original image')

# Plot the second image on the right subplot
axs[1].imshow(blended,cmap='gray')
axs[1].set_title('Blended result')

# Show the figure
plt.show()

"""

from PIL import Image

def get_y_and_y_pred(df):
    
    mask_vectors = []
    mask_vectors_blended = []
    mask_vectors_blended_prob = []
    for index, row in df.iterrows():
        image_path = row['PATH_TO_ORIGINAL_IMAGE']  
        mask_path = row['MASK'] 
        
        try:
            image = cv2.imread(image_path)
            mask = Image.open(mask_path).convert('L')
            
            image_vector = np.array(image) 
            mask = np.array(mask)
            mask_binary = np.where(mask > 128, 1, 0)
            #.flatten()  # Flatten image into a 1D vector
            mask_vectors.append(mask_binary.flatten().tolist())
            
            predictions_smooth = predict_img_with_smooth_windowing(
                    image_vector,
                    window_size = 128,
                    subdivisions = 2,
                    nb_classes = 2,
                    pred_func = (
                        #lambda img_bath_subdiv: np.argmax (nn_unet_vgg16.predict(img_bath_subdiv), axis = -1) 
                        lambda img_bath_subdiv: nn_unet_vgg16.predict(img_bath_subdiv)
                        #predictImg(nn_unet_vgg16, img)[1]
                        #lambda img: predict(nn_unet_vgg16, img)[1]
                        )
                    )

            final_prediction = np.argmax(predictions_smooth, axis = 2)
            
            # for auc
            f_0 = predictions_smooth[:,:,0].flatten()
            f_1 = predictions_smooth[:,:,1].flatten()
            f_final = final_prediction.flatten()
            
            probability = []
            for indx, value in enumerate(f_final):
                if value == 0:
                    probability.append(1-f_0[indx])
                if value == 1:
                    probability.append(f_1[indx])
                    
            mask_vectors_blended_prob.append(probability)
            
            final_prediction = (final_prediction).astype(np.uint8)
            mask_vectors_blended.append(final_prediction.flatten().tolist())   
   
        except Exception as e:
            print(f"Error reading image at path {image_path}: {str(e)}")

    
    
    
    return image_vector, mask_vectors, mask_vectors_blended, mask_vectors_blended_prob
    
img, mask_vectors, mask_vectors_blended, mask_vectors_blended_prob = get_y_and_y_pred(df_drive)

from itertools import chain
y_true = list(chain(*mask_vectors))
y_pred = list(chain(*mask_vectors_blended))
y_prob = list(chain(*mask_vectors_blended_prob))

from sklearn.metrics import confusion_matrix, roc_curve, auc
cm = confusion_matrix(y_true, y_pred, normalize=False)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred ).ravel()
print(tn, fp, fn, tp )


def report(cmat: dict):

    Np = cmat['TP'] + cmat['FP']
    Nn = cmat['TN'] + cmat['FN']

    accuracy = (float) (cmat['TP'] + cmat['TN']) / (float) (sum(cmat.values()))

    iou = {}
    iou['N'] = (float) (cmat['TN']) / (float) (cmat['TN'] + cmat['FN'] + cmat['FP'])
    iou['P'] = (float) (cmat['TP']) / (float) (cmat['TP'] + cmat['FP'] + cmat['FN'])
    iou['avr'] = (iou['N'] + iou['P']) / 2.
    iou['avr_weighted'] = (Nn * iou['N'] + Np * iou['P']) / (Nn + Np)

    precision = {}
    precision['N'] =  (float) (cmat['TN']) / (float) (cmat['TN'] + cmat['FN'])
    precision['P'] =  (float) (cmat['TP']) / (float) (cmat['TP'] + cmat['FP'])
    precision['avr'] = (precision['N'] + precision['P']) / 2.
    precision['avr_weighted'] =  (Nn * precision['N'] + Np * precision['P']) / (Nn + Np)

    recall = {}
    recall['N'] =  (float) (cmat['TN']) / (float) (cmat['TN'] + cmat['FP'])
    recall['P'] =  (float) (cmat['TP']) / (float) (cmat['TP'] + cmat['FN'])
    recall['avr'] =  (recall['N'] + recall['P']) / 2.
    recall['avr_weighted'] =  (Nn * recall['N'] + Np * recall['P']) / (Nn + Np)

    specifity = {}
    specifity['N'] = recall['P']
    specifity['P'] = recall['N']
    specifity['avr'] =  (specifity['N'] + specifity['P']) / 2.
    specifity['avr_weighted'] =  (Nn * specifity['N'] + Np * specifity['P']) / (Nn + Np)

    F1 = {}
    F1['N'] =  2. * (precision['N'] * recall['N']) / (float) (precision['N'] + recall['N'])
    F1['P'] =  2. * (precision['P'] * recall['P']) / (float) (precision['P'] + recall['P'])
    F1['avr'] =  (F1['N'] + F1['P']) / 2.
    F1['avr_weighted'] =  (Nn * F1['N'] + Np * F1['P']) / (Np + Nn)

    print('accuracy: ', accuracy)
    print('precision: ', precision)
    print('recall: ', recall)
    print('specifity: ', specifity)
    print('F1: ', F1)
    print('iou: ', iou)
    
np_cmat = np.array([[22413731, 394812], [435465, 1889048]])
np_cmat = np.array([[5952799,   76786], [130442,  439173]]) # DRIVE
print(np_cmat)
cmat = {'TN': np_cmat[0,0], 'FN': np_cmat[0,1], 'FP': np_cmat[1,0], 'TP': np_cmat[1,1]}
report(cmat)    




def plot_roc(y_true, y_pred):
    # calculate the fpr and tpr for all thresholds of the classification
    predictions = y_pred
    fpr, tpr, threshold = roc_curve(y_true, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))  # Increase the figure size
    plt.title('Receiver Operating Characteristic', fontsize=16)  # Increase the title font size
    plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc, linewidth=2)  # Increase the line width and label font size
    plt.legend(loc='lower right', fontsize=14)  # Increase the legend font size
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1.5)  # Increase the line width
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize=14)  # Increase the y-axis label font size
    plt.xlabel('False Positive Rate', fontsize=14)  # Increase the x-axis label font size
    plt.tick_params(axis='both', labelsize=12)  # Increase the tick label font size
    plt.show()

    
plot_roc(y_true, y_prob)
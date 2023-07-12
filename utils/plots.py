import numpy as np
import pandas as pd

import cv2 as opencv
import matplotlib.pylab as plt
from matplotlib.widgets import Slider, RangeSlider
import seaborn as sns
from PIL import Image


def imshow(src: np.ndarray, ax=None, title: str = None, figsize: tuple = None, to_bgra: bool = True) -> None:

    # matplot lib works with rgb order of channel so input image needs conversion
    src_rgb = opencv.cvtColor(src, opencv.COLOR_BGR2RGB) if to_bgra and len(src.shape) == 3 else src

    if ax is not None:
        ax.imshow(src_rgb)
        ax.axis('off')
        if title is not None: ax.set_title(title)
    else:
        if figsize is not None: plt.rcParams['figure.figsize'] = figsize
        plt.imshow(src_rgb)
        plt.axis('off')
        if title is not None: plt.title(title)


def maskshow(mask: np.ndarray, ax=None, title: str = None, figsize: tuple = None) -> None:

    cmap_str = 'gray'

    if ax is not None:
        ax.imshow(mask, cmap=cmap_str)
        ax.axis('off')
        if title is not None: ax.set_title(title)
    else:
        if figsize is not None: plt.rcParams['figure.figsize'] = figsize
        plt.imshow(mask, cmap=cmap_str)
        plt.axis('off')
        if title is not None: plt.title(title)


def plotTrainingHistory(history: pd.DataFrame) -> None:

    fig, ax = plt.subplots(1, 2, figsize=(20, 4))

    # plot training and validation loss
    sns.lineplot(x=history.index + 1, y='loss', data=history, ax=ax[0])
    fig1 = sns.lineplot(x=history.index + 1, y='val_loss', data=history, ax=ax[0])

    fig1.set_xlabel('#epoch', fontsize=18)
    fig1.set_ylabel('Loss (log scale)', fontsize=18)
    fig1.tick_params(axis='both', which='major', labelsize=14)
    fig1.set(yscale='log')

    # plot training and validation IoU
    sns.lineplot(x=history.index + 1, y='mean_io_u', data=history, ax=ax[1])
    fig2 = sns.lineplot(x=history.index + 1, y='val_mean_io_u', data=history, ax=ax[1])

    fig2.set_xlabel('#epoch', fontsize=18)
    fig2.set_ylabel('Mean IoU', fontsize=18)
    fig2.tick_params(axis='both', which='major', labelsize=14)

    plt.legend(labels=['Training', 'Validation'], loc='lower center', bbox_to_anchor=(-0.1, -0.4), prop={'size': 14})
    plt.show()


def convertProbability(img_prob: np.ndarray, img_labels: np.ndarray) -> np.ndarray:
    # convert to P(X | y = 1)
    return np.abs(np.ones(img_labels.shape, dtype=np.float32) - img_labels - img_prob)


def plotColorizedVessels(fn_img: str, predictImg, nn_model, blended = np.zeros([1,1])) -> None:
    
    img = opencv.imread(fn_img, opencv.IMREAD_COLOR)
    img = opencv.cvtColor(img, opencv.COLOR_BGR2RGB)
    
    if img.shape[:2] == blended.shape:
        vessels = opencv.bitwise_or(img, img, mask=blended) 
    else:
        predicted_prob, predicted_label = predictImg(nn_model, img)
        vessels = opencv.bitwise_or(img, img, mask=predicted_label)     
       
    # Combine the two images using weighted addition - manual parameters
    #comb = opencv.addWeighted(img, 1-alpha, vessels, alpha, 0)
    comb = opencv.addWeighted(img, 0.95, vessels, 1, 0)   
    
    plt.figure(figsize=(10,10))
    plt.imshow(comb)
    plt.title('Colorized mask in original image')
    plt.show()
    

def plotPredictedImg(fn_img: str, fn_label: str, predictImg, nn_model) -> None:

    img = opencv.imread(fn_img, opencv.IMREAD_COLOR)

    pil_img = Image.open(fn_label)
    label = np.array(pil_img) / 255; label = label.astype(np.uint8)

    predicted_prob, predicted_label = predictImg(nn_model, img)
    predicted_prob = convertProbability(predicted_prob, predicted_label)

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
    
    
def plotPredictedImg2x2(fn_img: str, fn_label: str, predictImg, nn_model) -> None:

    img = opencv.imread(fn_img, opencv.IMREAD_COLOR)

    pil_img = Image.open(fn_label)
    label = np.array(pil_img) / 255; label = label.astype(np.uint8)

    predicted_prob, predicted_label = predictImg(nn_model, img)
    predicted_prob = convertProbability(predicted_prob, predicted_label)

    mask_titles = ['Mask (true)', 'Mask (pred. probability)', 'Mask (pred. label)']
    mask_imgs = [label, predicted_prob, predicted_label]

    fig, axes = plt.subplots(2, 2, figsize=(10, 3))
    # plot source image
    imshow(img, ax=axes[0,0], title='Input image')
    from itertools import product
    # plot mask
    for mask, title, idx in zip(mask_imgs, mask_titles, list(product([0, 1], repeat=2))[1:]):
        maskshow(mask, ax=axes[idx], title=title)
    fig.suptitle('Image {}'.format(fn_img))
    plt.show()

    
def plotPredictedImgSlicer(fn_img: str, fn_label: str, predictImg, nn_model):

    img = opencv.imread(fn_img, opencv.IMREAD_COLOR)
    predicted_prob, predicted_label = predictImg(nn_model, img)
    predicted_prob = convertProbability(predicted_prob, predicted_label)
    init_prob = np.mean(predicted_prob)

    # visualize
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(bottom=0.25)
    
    im = axs[0].imshow(img)
    axs[1].imshow(predicted_prob)
    axs[1].set_title('Pixels above given probability')
    
    # Create the RangeSlider
    slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03]) # recttuple (left, bottom, width, height)
    slider = Slider(
        ax=slider_ax,
        label='Probability',
        valmin=predicted_prob.min(),
        valmax=predicted_prob.max(),
        valinit=init_prob,
    )

    def update(label_probability):
        # The val passed to a callback by the RangeSlider will
        # be a tuple of (min, max)
        print(label_probability)
        # Update the image's colormap
        # does not work
        # (thresh, im_bw) = opencv.threshold(predicted_prob, label_probability, 1, opencv.THRESH_BINARY)
        im_bw = predicted_prob.copy()
        im_bw[im_bw <= label_probability] = 0

        axs[0].imshow(img)
        axs[1].imshow(im_bw,  cmap='gray')
    
        # Redraw the figure to ensure it updates
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show() 
    
    return fig, slider_ax, slider  # https://github.com/matplotlib/matplotlib/issues/3105/


def plotHistogramImgSlicer(fn_img: str, fn_label: str, predictImg, nn_model):

    img = opencv.imread(fn_img, opencv.IMREAD_COLOR)
    img = opencv.cvtColor(img, opencv.COLOR_BGR2RGB)
    predicted_prob, predicted_label = predictImg(nn_model, img)
    predicted_prob = convertProbability(predicted_prob, predicted_label)

    # visualize
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    fig.subplots_adjust(bottom=0.25)
    
    im = axs[0].imshow(predicted_prob, cmap='gray')
    n = axs[1].hist(predicted_prob.flatten(), bins='auto')
    print(np.mean(n[0]))
    axs[1].set_title('Histogram of pixel intensities')
    axs[1].set_ylim([0,3000])
    
    # Create the RangeSlider
    slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03]) # recttuple (left, bottom, width, height)
    slider = RangeSlider(slider_ax, "Threshold", predicted_prob.min(), predicted_prob.max())
    
    # Create the Vertical lines on the histogram
    lower_limit_line = axs[1].axvline(slider.val[0], color='k')
    upper_limit_line = axs[1].axvline(slider.val[1], color='k')

    def update(val):
        # The val passed to a callback by the RangeSlider will
        # be a tuple of (min, max)
    
        # Update the image's colormap
        im.norm.vmin = val[0]
        im.norm.vmax = val[1]
    
        # Update the position of the vertical lines
        lower_limit_line.set_xdata([val[0], val[0]])
        upper_limit_line.set_xdata([val[1], val[1]])
    
        # Redraw the figure to ensure it updates
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show() 
    
    return fig, slider_ax, slider  # https://github.com/matplotlib/matplotlib/issues/3105/

def clean_image(img_gray: np.ndarray, percentage:float = 5e-4, prob_threshold:float = 0.5,  plot:bool = False):
    """ 
    function to remove small unconected white spaces from 
    mask image 
    Attribute:
        binary_threshold - the level of gray from which the image is separated into 
            two domains
        percentage - all areas with size smaller than some percent 
        of image are deleted
        prob_threshold : in the case of mask_prob visualize only pixels with 
            probabilty higher than threshold
        plot - Visualize the result
    """   
    
    # convert to binary by thresholding, due to np.argmax mask_image is binary, e.g., 
    # contains only 0-black, 255-white vessels, but it has to multiplied by 255 and 
    # retype to np.unit8, otherwise cv2 does not accept it 
    # !!! This threshold is works only with probability label (mask_prob) which is in 
    # interval 0-255, binary label (mask_label) is only zero or 255 - in this case
    # the prob_threshold does not change anything, but we do not have to 
    # write two functions
    threshold = int(255 * prob_threshold)
    ret, binary_map = opencv.threshold(img_gray, threshold,255, opencv.THRESH_BINARY)

    # find connected components
    nlabels, labels, stats, centroids = opencv.connectedComponentsWithStats(binary_map, 8, opencv.CV_32S)

    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,opencv.CC_STAT_AREA]
    
    # treshold is the size in pixels and it is percent of image
    treshold = round(img_gray.shape[0]*img_gray.shape[1]*percentage)

    result = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        #print(areas[i])
        if areas[i] >= treshold:   #keep
            result[labels == i + 1] = 1
    
    if plot:
        import matplotlib.pyplot as plt
        # Create a figure and axis
        fig, ax = plt.subplots(1, 2)

        # Plot the first image
        ax[0].imshow(binary_map)
        ax[0].axis('off')
        ax[0].set_title("Binary image - Original", fontsize=14)

        # Plot the second image
        ax[1].imshow(result)
        ax[1].axis('off')
        ax[1].set_title(f"Result for treshold {treshold}", fontsize=14)

        # Show the plot
        plt.show()
    return result

def plotListofImages(predictions: dict[str, dict[np.ndarray,np.ndarray, np.ndarray]], clean_threshold = 0, prob_threshold=0.7) -> None:
    """
    Plots the dict of predicted images
    Parameters
    ----------
    preditions : dict from inference.predictListOfFiles
    clean_threshold : all areas with size smaller than some percent 
        of image are deleted
    prob_threshold : in the case of mask_prob visualize only pixels with 
        probabilty higher than threshold
        
    For more details about parameters see clean_image()
        
    Returns
    -------
    Visualization - original visualalisation or cleaned images     
    """
       
    for key, item in predictions.items(): 
        fig, axs = plt.subplots(nrows=1,ncols=3)
        # plot original image
        plt.sca(axs[0]); 
        plt.imshow(opencv.cvtColor(item['image'], opencv.COLOR_BGR2RGB)); plt.title('Original image')
        # plot mask - probability
        if clean_threshold:
            img2 = item['mask_prob'] * 255
            img2 = img2.astype(np.uint8)
            plt.sca(axs[1]) 
            plt.imshow(clean_image(img2, percentage=clean_threshold, prob_threshold=prob_threshold))            
            plt.title(f'Mask - probability. \n Treshold (area to keep) {clean_threshold},\n Probability to draw {prob_threshold}')
        else:        
            plt.sca(axs[1]) 
            plt.imshow(item['mask_prob'])
            plt.title('Mask - probability')
        # mask label
        if clean_threshold:
            img3 = item['mask_label'] * 255
            img3 = img3.astype(np.uint8)
            plt.sca(axs[2]) 
            plt.imshow(clean_image(img3, percentage=clean_threshold)); 
            plt.title(f'Mask - probability.\n Treshold {clean_threshold}')
        else:        
            plt.sca(axs[2]) 
            plt.imshow(item['mask_label'])
            plt.title('Mask - label')
        plt.show()
        
    

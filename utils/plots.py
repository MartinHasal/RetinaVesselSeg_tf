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

    fig1.set_xlabel('#epoch', fontsize=14)
    fig1.set_ylabel('Loss (log scale)', fontsize=14)
    fig1.set(yscale='log')

    # plot training and validation IoU
    sns.lineplot(x=history.index + 1, y='mean_io_u', data=history, ax=ax[1])
    fig2 = sns.lineplot(x=history.index + 1, y='val_mean_io_u', data=history, ax=ax[1])

    fig2.set_xlabel('#epoch', fontsize=14)
    fig2.set_ylabel('Mean IoU', fontsize=14)

    plt.legend(labels=['Training', 'Validation'], loc='lower center', bbox_to_anchor=(-0.1, -0.4), prop={'size': 14})
    plt.show()


def convertProbability(img_prob: np.ndarray, img_labels: np.ndarray) -> np.ndarray:
    # convert to P(X | y = 1)
    return np.abs(np.ones(img_labels.shape, dtype=np.float32) - img_labels - img_prob)

def plotColorizedVessels(fn_img: str, predictImg, nn_model) -> None:
    
    img = opencv.imread(fn_img, opencv.IMREAD_COLOR)
    img = opencv.cvtColor(img, opencv.COLOR_BGR2RGB)
    predicted_prob, predicted_label = predictImg(nn_model, img)
    
    plt.figure(figsize=(10,10))
    plt.imshow(opencv.bitwise_or(img, img, mask=predicted_label))
    plt.title('Colorized mask by original image')
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

    
def plotPredictedImgSlicer(fn_img: str, fn_label: str, predictImg, nn_model) -> None:    

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
    
    return fig, slider_ax, slider # https://github.com/matplotlib/matplotlib/issues/3105/



    
def plotHistogramImgSlicer(fn_img: str, fn_label: str, predictImg, nn_model) -> None:
    

    img = opencv.imread(fn_img, opencv.IMREAD_COLOR)
    img = opencv.cvtColor(img, opencv.COLOR_BGR2RGB) # ZEPTEJ SE MARYHO PROC DO PREDICTED JDE RGB???
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
    
    return fig, slider_ax, slider # https://github.com/matplotlib/matplotlib/issues/3105/   
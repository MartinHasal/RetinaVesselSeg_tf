import numpy as np
import pandas as pd

import cv2 as opencv
import matplotlib.pylab as plt
import seaborn as sns


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

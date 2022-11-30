import numpy as np

import cv2 as opencv
import matplotlib.pylab as plt


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

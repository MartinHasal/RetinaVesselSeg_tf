
import numpy as np
import tensorflow as tf

from typing import Union

from imblearn import metrics as imblearn_metrics
from matplotlib import pylab as plt_pylab
from sklearn import metrics as sklearn_metrics


def classification_report(labels_true: np.ndarray, labels_pred: np.ndarray) -> None:

    # print classification report and its imbalanced version
    print('Classification report:\n')
    print(sklearn_metrics.classification_report(
        labels_true[~np.isnan(labels_true)], labels_pred[~np.isnan(labels_true)]
    ))

    print('\nClassification report (imbalanced):\n')
    print(imblearn_metrics.classification_report_imbalanced(
        labels_true[~np.isnan(labels_true)], labels_pred[~np.isnan(labels_true)]
    ))

    """
    computing IoU (Intersection over Union) for positive samples
    """

    iou_score_p = sklearn_metrics.jaccard_score(y_true=labels_true[~np.isnan(labels_true)],
                                                y_pred=labels_pred[~np.isnan(labels_true)])
    """
    computing IoU  for negative samples
    """

    labels_true = labels_true[~np.isnan(labels_true)]
    labels_pred = labels_pred[~np.isnan(labels_pred)]

    min = labels_true.min(); max = labels_true.max()

    if min != 0:
        labels_true[labels_true == min] = 0
        labels_pred[labels_pred == min] = 0
    if max != 1:
        labels_true[labels_true == max] = 1
        labels_pred[labels_pred == max] = 1

    labels_true = labels_true.astype(np.int32)
    labels_true = np.where((labels_true == 0) | (labels_true == 1), labels_true ^ 1, labels_true)

    labels_pred = labels_pred.astype(np.int32)
    labels_pred = np.where((labels_pred == 0) | (labels_pred == 1), labels_pred ^ 1, labels_pred)

    iou_score_n = sklearn_metrics.jaccard_score(y_true=labels_true, y_pred=labels_pred)

    # computing mean intersection over union
    iou_score_m = (iou_score_p + iou_score_n) / 2.

    print('\nIoU- (intersection over union): {:.2f}'.format(iou_score_n))
    print('\nIoU+ (intersection over union): {:.2f}'.format(iou_score_p))
    print('\nmIoU (mean IoU): {:.2f}'.format(iou_score_m))


def plot_aucroc(labels_true: np.ndarray, labels_pred: np.ndarray, ax=None) -> None:

    from utils import roc

    auc_roc = roc.AucRoc(y_true=labels_true, y_pred=labels_pred)
    auc_roc.plot(ax=ax)


def plot_cmat(labels_true: np.ndarray, labels_pred: np.ndarray, ax=None) -> None:

    from utils import cmat

    cm = cmat.ConfusionMatrix(
        y_true=labels_true[~np.isnan(labels_true)],
        y_pred=labels_pred[~np.isnan(labels_true)],
        labels=['Background', 'Vessel']
    )

    cm.plot(ax=ax)


def show_report(test: Union[np.ndarray, tf.data.Dataset],
                labels_pred: np.ndarray,
                with_aucroc: bool = False,
                with_cmat: bool = False,
                with_report: bool = False) -> None:

    if isinstance(test, tf.data.Dataset):
        labels_true = np.concatenate([y for _, y in test], axis=0).reshape(-1)
    elif isinstance(test, np.ndarray):
        labels_true = test.reshape(-1)
    else:
        raise NotImplementedError

    labels_pred = labels_pred.reshape(-1)

    if with_report:

        classification_report(labels_true=labels_true, labels_pred=labels_pred)

    if with_cmat and with_aucroc:

        if with_report: print('\n')

        _, axes = plt_pylab.subplots(1, 2, figsize=(10, 5))

        plot_aucroc(labels_true=labels_true, labels_pred=labels_pred, ax=axes[0])
        plot_cmat(labels_true=labels_true, labels_pred=labels_pred, ax=axes[1])

    elif with_cmat:

        plot_cmat(labels_true=labels_true, labels_pred=labels_pred)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class ConfusionMatrix(object):

    def __init__(self,
                 tst: np.ndarray,
                 y_pred: np.ndarray,
                 labels=None,
                 normalize: bool = True):

        if isinstance(tst, tf.data.Dataset):
            self._y_true = np.concatenate([y for _, y in tst], axis=0).reshape(-1)
        else:
            self._y_true = tst.reshape(-1)

        self._y_pred = y_pred.reshape(-1)
        self._labels = labels

        self._cm = None
        self._cm_normalize = normalize

    def __computeConfusionMatrix(self) -> None:

        if self._cm is not None:
            return

        self._cm = confusion_matrix(
            self._y_true,
            self._y_pred,
            normalize='true' if self._cm_normalize else None
        )
    
    def get_cm(self):
        self.__computeConfusionMatrix()
        return self._cm

    def plot(self,
             figsize: tuple = (25, 20),
             label_fontsize: int = 20,
             ticks_fontsize: int = 15,
             title: str = 'Confusion matrix',
             title_fontsize: int = 25,
             value_size: int = 25
             ) -> None:

        self.__computeConfusionMatrix()

        disp = ConfusionMatrixDisplay(
            confusion_matrix=self._cm,
            display_labels=self._labels,
        )

        fig, ax = plt.subplots(figsize=figsize)

        font = {'size': value_size}
        plt.rc('font', **font)

        disp.plot(cmap='BuPu', ax=ax)

        plt.title(title, fontsize=title_fontsize)

        plt.xlabel('Predicted label', fontsize=label_fontsize)
        plt.ylabel('True label', fontsize=label_fontsize)

        plt.xticks(fontsize=ticks_fontsize, rotation=90)
        plt.yticks(fontsize=ticks_fontsize)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    y_true = np.asarray([1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 7, 8, 9, 10])
    y_pred = np.asarray([2, 2, 3, 10, 4, 4, 5, 5, 6, 7, 6, 7, 8, 9, 10])

    labels = ['aaaaaa', 'bbbbbb', 'cccccc', 'dddd', 'eeeee', 'fffff', 'ggggg', 'hhhhhh', 'iiiiiii', 'jjjjjj']

    cm = ConfusionMatrix(
        y_true,
        y_pred,
        labels,
        normalize=False
    )

    cm.plot(figsize=(25, 20), title_fontsize=50, label_fontsize=30, ticks_fontsize=25)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class ConfusionMatrix(object):

    def __init__(self,
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 labels=None,
                 normalize: bool = True):

        self._y_true = y_true
        self._y_pred = y_pred
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

    def plot(self,
             title: str = 'Confusion matrix',
             ax=None) -> None:

        self.__computeConfusionMatrix()

        disp = ConfusionMatrixDisplay(
            confusion_matrix=self._cm,
            display_labels=self._labels,
        )

        disp.plot(cmap='BuPu', ax=ax)
        plt.title(title)

        plt.xlabel('Predicted label')
        plt.ylabel('True label')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    y_true = np.asarray([1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 7, 8, 9, 10])
    y_pred = np.asarray([2, 2, 3, 10, 4, 4, 5, 5, 6, 7, 6, 7, 8, 9, 10])

    labels = ['aaaaaa', 'bbbbbb', 'cccccc', 'dddd', 'eeeee', 'fffff', 'ggggg', 'hhhhhh', 'iiiiiii', 'jjjjjj']

    cm = ConfusionMatrix(
        y_true,
        y_pred,
        labels
    )

    cm.plot()
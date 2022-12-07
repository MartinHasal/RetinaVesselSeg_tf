import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc


class AucRoc(object):

    def __init__(self, y_true, y_pred):

        self._y_true = y_true
        self._y_pred = y_pred

        self._fpr = None
        self._tpr = None

        self._auc = None

    @property
    def auc(self) -> float:

        if self._auc is None:
            self.__compute()

        return self._auc

    @property
    def fpr(self) -> float:

        if self._fpr is None:
            self.__compute()

        return self._fpr

    @property
    def tpr(self) -> float:

        if self._tpr is None:
            self.__compute()

        return self._tpr

    def __reset(self) -> None:

        del self._fpr, self._tpr
        self._fpr = self._tpr = None

        self._auc = None

    def __compute(self) -> None:

        self._fpr, self._tpr, _ = roc_curve(self._y_true.reshape(-1), self._y_pred.reshape(-1))
        self._auc = auc(self._fpr, self._tpr)

    def plot(self) -> None:

        fpr = self.fpr
        tpr = self.tpr

        plt.plot(
            fpr,
            tpr,
            color='darkorange',
            label='AUC ROC = {0:.4f}'.format(self._auc),
        )
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.title('Receiver operating characteristic')
        plt.legend(loc='lower right')

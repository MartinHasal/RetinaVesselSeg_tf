import tensorflow as tf

from keras.metrics import MeanIoU as KerasMeanIoU


class FixedMeanIoU(KerasMeanIoU):

    def __init__(self,
                 y_true=None,
                 y_pred=None,
                 num_classes=None,
                 name=None,
                 dtype=None):

        super(FixedMeanIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self,
                     y_true,
                     y_pred,
                     sample_weight=None):

        y_pred = tf.math.argmax(y_pred, axis=-1)

        return super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):

        base_config = super().get_config()

        dict_config = {
            **base_config,
            'num_classes': self.num_classes
        }

        return dict_config

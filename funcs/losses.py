import tensorflow as tf

_EPSILON = tf.keras.backend.epsilon()


def fn_focal_loss(y_true, y_pred, gamma: float, class_weights=None) -> tf.Tensor:

    y_true = tf.dtypes.cast(y_true, dtype=tf.dtypes.int64)
    probs = y_pred

    logits = tf.math.log(tf.clip_by_value(y_pred, _EPSILON, 1 - _EPSILON))

    # compute cross entropy
    ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)

    y_true_rank = y_true.shape.rank
    # select computed probability for given category
    probs = tf.gather(probs, y_true, axis=-1, batch_dims=y_true_rank)

    # compute focal loss
    focal_loss = tf.math.pow(1. - probs, gamma) * ce_loss

    if class_weights is not None:
        class_weights = tf.convert_to_tensor(class_weights, dtype=tf.dtypes.float32)
        class_weights = tf.gather(class_weights, y_true, axis=0, batch_dims=y_true_rank)
        focal_loss *= class_weights

    return focal_loss


class FocalLoss(tf.keras.losses.Loss):

    def __init__(self,
                 gamma: float,
                 class_weights=None,
                 **kwargs):

        super().__init__(**kwargs)

        self._gamma = gamma

        self._class_weights = class_weights

    def get_config(self):

        base_config = super().get_config()

        dict_config = {
            **base_config,
            'gamma': self._gamma,
            'class_weights': self._class_weights
        }

        return dict_config

    def call(self, y_true, y_pred):

        return fn_focal_loss(y_true, y_pred, self._gamma, self._class_weights)

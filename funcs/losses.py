import tensorflow as tf

from enum import Enum

from tensorflow import keras
from tensorflow.keras import backend


class LossType(Enum):
    CROSS_ENTROPY = 'cross_entropy'
    DICE = 'dice'
    FOCAL_LOSS = 'focal'

    def __str__(self):
        return self.value


# focal loss implementation

_EPSILON = tf.keras.backend.epsilon()


def fn_focal_loss(y_true, y_pred, gamma: float, class_weights=None) -> tf.Tensor:
    y_true = tf.dtypes.cast(y_true, dtype=tf.dtypes.int64)
    y_pred = tf.convert_to_tensor(y_pred)

    # check if reshape is needed
    y_true_rank = y_true.shape.rank
    y_pred_rank = y_pred.shape.rank
    y_pred_shape = tf.shape(y_pred)

    reshape_needed = (y_true_rank is not None and y_pred_rank is not None and
                      y_pred_rank != y_true_rank + 1)
    if reshape_needed:
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1, y_pred_shape[-1]])

    # compute cross entropy
    logits = tf.math.log(tf.clip_by_value(y_pred, _EPSILON, 1 - _EPSILON))
    ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)

    y_true_rank = y_true.shape.rank
    # select computed probability for given category
    probs = tf.gather(y_pred, y_true, axis=-1, batch_dims=y_true_rank)

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


# Dice loss implementation


def gather_channels(*xs, indexes=None):
    if indexes is None:
        return xs
    elif isinstance(indexes, (int)):
        indexes = [indexes]

    xs_ = []
    for x in xs:
        if backend.image_data_format() == "channels_last":
            x = tf.gather(x, indexes, axis=-1)
        else:
            x = tf.gather(x, indexes, axis=1)
        xs_.append(x)

    return xs_


def get_reduce_axes(per_image):
    axes = [1, 2] if backend.image_data_format() == "channels_last" else [2, 3]
    if not per_image:
        axes.insert(0, 0)

    return axes


class DiceBinaryLoss(tf.keras.losses.Loss):

    def __init__(self,
                 beta=1,
                 class_ids=None,
                 per_image=False,
                 smooth=_EPSILON,
                 label_smoothing=0.0,
                 from_logits=False,
                 class_indexes=None,
                 **kwargs):

        super().__init__(**kwargs)

        self.beta = beta
        self.class_ids = class_ids
        self.per_image = per_image
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.smooth = smooth
        self.class_indexes = class_indexes

    def get_config(self):
        base_config = super().get_config()

        dict_config = {
            **base_config,
            'gamma': self._gamma,
            'class_weights': self._class_weights
        }

        return dict_config

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        label_smoothing = tf.convert_to_tensor(self.label_smoothing, dtype=y_pred.dtype)

        y_pred = tf.__internal__.smart_cond.smart_cond(
            self.from_logits, lambda: tf.nn.softmax(y_pred), lambda: y_pred
        )

        def _smooth_labels():
            num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
            return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

        y_true = tf.__internal__.smart_cond.smart_cond(
            label_smoothing, _smooth_labels, lambda: y_true
        )

        y_true, y_pred = tf.__internal__.smart_cond.smart_cond(
            self.class_indexes == None,
            lambda: (y_true, y_pred),
            lambda: gather_channels(y_true, y_pred, indexes=self.class_ids),
        )

        axes = get_reduce_axes(self.per_image)

        true_positive = keras.backend.sum(y_true * y_pred, axis=axes)
        false_positive = keras.backend.sum(y_pred, axis=axes) - true_positive
        false_negative = keras.backend.sum(y_true, axis=axes) - true_positive

        # Type I and type II errors - f-score forumula
        power_beta = 1 + self.beta ** 2
        numerator = power_beta * true_positive + self.smooth
        denominator = (
            (power_beta * true_positive)
            + (self.beta ** 2 * false_negative)
            + false_positive
            + self.smooth
        )

        dice_score = numerator / denominator
        dice_score = tf.cond(
            tf.constant(self.per_image, dtype=tf.bool),
            lambda: keras.backend.mean(dice_score, axis=0),
            lambda: keras.backend.mean(dice_score),
        )

        return 1 - dice_score

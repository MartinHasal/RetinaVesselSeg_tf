import tensorflow as tf

from keras import callbacks as KerasCallbacks
from keras.engine.functional import Functional as KerasFunctional

from tensorflow.keras import optimizers as KerasOptimizers
from tensorflow.keras.optimizers import schedules as KerasSchedules

from funcs.losses import FocalLoss
from funcs.metrics import FixedMeanIoU


def trainSegmentationModel(nn_model: KerasFunctional,
                           nclasses: int,
                           ds_train: tf.data.Dataset,
                           ds_val: tf.data.Dataset = None,
                           nepochs: int = 5,
                           loss_type: str = 'cross_entropy',
                           batch_size: int = 32,
                           buffer_size: int = 1000,
                           focal_loss_gamma: float = 2.,
                           class_weights=None):

    if loss_type not in ['cross_entropy', 'focal_loss']:
        raise ValueError('Supported loss types are cross entropy and focal loss.')

    # set loss types
    if loss_type == 'cross_entropy':
        fn_loss = 'sparse_categorical_crossentropy'
    else:
        fn_loss = FocalLoss(gamma=focal_loss_gamma, class_weights=class_weights)

    # set performance metric
    mean_io_u = FixedMeanIoU(num_classes=nclasses, name='mean_io_u')
    metrics = [mean_io_u]

    # set learning rate decay
    initial_learning_rate = .01
    decay_steps = 10000
    decay_rate = .96

    lr_schedule = KerasSchedules.ExponentialDecay(initial_learning_rate,
                                                  decay_steps=decay_steps,
                                                  decay_rate=decay_rate,
                                                  staircase=True)

    # set optimizer and compile model
    optimizer = KerasOptimizers.Adam(learning_rate=lr_schedule)
    nn_model.compile(optimizer=optimizer, loss=fn_loss, weighted_metrics=metrics)

    # set call backs
    early_stopping = KerasCallbacks.EarlyStopping(monitor='val_mean_io_u',
                                                  patience=10,
                                                  restore_best_weights=True,
                                                  mode='max')

    # process training and validation dataset
    train_batches = (
        ds_train.cache()
        .shuffle(buffer_size, seed=42)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        .repeat(2)
    )

    val_batches = (
        ds_val
        .cache()
        .shuffle(buffer_size, seed=42)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # train model
    history = nn_model.fit(train_batches,
                           epochs=nepochs,
                           validation_split=0.2 if ds_val is None else 0,
                           validation_data=val_batches,
                           verbose=1,
                           shuffle=True,
                           callbacks=[early_stopping])

    return history
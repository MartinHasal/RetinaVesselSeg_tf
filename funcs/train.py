import tensorflow as tf

from funcs.lr import LearningRateDecayType, ScheduleWarmupExponentialDecay

from keras import callbacks as KerasCallbacks
from keras.engine.functional import Functional as KerasFunctional

from tensorflow.keras import optimizers as KerasOptimizers
from tensorflow.keras.optimizers import schedules as KerasSchedules

from funcs.losses import LossType, FocalLoss, DiceBinaryLoss
from funcs.metrics import FixedMeanIoU


def trainSegmentationModel(nn_model: KerasFunctional,
                           nclasses: int,
                           ds_train: tf.data.Dataset,
                           ds_val: tf.data.Dataset = None,
                           nepochs: int = 10,
                           loss_type: LossType = LossType.CROSS_ENTROPY,
                           batch_size: int = 32,
                           buffer_size: int = 1000,
                           decay: LearningRateDecayType = LearningRateDecayType.EXPONENTIAL,
                           focal_loss_gamma: float = 2.,
                           display_callback: bool = False,
                           warmup_decay_info: bool = False,
                           class_weights=None):

    # set loss types
    if loss_type == LossType.CROSS_ENTROPY:
        fn_loss = 'sparse_categorical_crossentropy'
    elif loss_type == LossType.DICE:
        fn_loss = DiceBinaryLoss()
    else:
        fn_loss = FocalLoss(gamma=focal_loss_gamma, class_weights=class_weights)

    # set performance metric
    mean_io_u = FixedMeanIoU(num_classes=nclasses, name='mean_io_u')
    metrics = [mean_io_u]

    # set learning rate decay    
    if decay == LearningRateDecayType.EXPONENTIAL:
        initial_learning_rate = .01
        decay_steps = 10000
        decay_rate = .96
    
        lr_schedule = KerasSchedules.ExponentialDecay(initial_learning_rate,
                                                      decay_steps=decay_steps,
                                                      decay_rate=decay_rate,
                                                      staircase=True)
    else:
        lr_schedule = ScheduleWarmupExponentialDecay(nepochs=nepochs, info=warmup_decay_info)

    # set optimizer and compile model
    optimizer = KerasOptimizers.Adam(learning_rate=lr_schedule)
    nn_model.compile(optimizer=optimizer, loss=fn_loss, weighted_metrics=metrics)

    # set call backs
    early_stopping = KerasCallbacks.EarlyStopping(monitor='val_mean_io_u',
                                                  patience=10,
                                                  restore_best_weights=True,
                                                  mode='max')
                                                  
    # display prediction online
    """
    class DisplayCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs=None):
        if epoch % (nepochs / 5) == 0: # display evolution of algorithm every 5th epoch 
          # clear_output(wait=True) # if you want replace the images each time, uncomment this
          show_predictions(train_dataset, 7)
          print ('\nSample Prediction after epoch {}\n'.format(epoch+1))                                     
    """

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

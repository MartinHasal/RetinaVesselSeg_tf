import tensorflow as tf

from keras import callbacks as KerasCallbacks
from keras.engine.functional import Functional as KerasFunctional

from tensorflow.keras import optimizers as KerasOptimizers
from tensorflow.keras.optimizers import schedules as KerasSchedules

from funcs.losses import FocalLoss, DiceBinaryLoss
from funcs.metrics import FixedMeanIoU


def trainSegmentationModel(nn_model: KerasFunctional,
                           nclasses: int,
                           ds_train: tf.data.Dataset,
                           ds_val: tf.data.Dataset = None,
                           nepochs: int = 10,
                           loss_type: str = 'cross_entropy',
                           batch_size: int = 32,
                           buffer_size: int = 1000,
                           decay = 'exponential',
                           focal_loss_gamma: float = 2.,
                           display_callback = False,
                           class_weights=None):

    if loss_type not in ['cross_entropy', 'focal_loss', 'dice']:
        raise ValueError('Supported loss types are cross entropy and focal loss.')

    # set loss types
    if loss_type == 'cross_entropy':
        fn_loss = 'sparse_categorical_crossentropy'
    elif loss_type == 'dice':
        fn_loss = DiceBinaryLoss()
    else:
        fn_loss = FocalLoss(gamma=focal_loss_gamma, class_weights=class_weights)

    # set performance metric
    mean_io_u = FixedMeanIoU(num_classes=nclasses, name='mean_io_u')
    metrics = [mean_io_u]


    
    # set learning rate decay    
    if decay == 'exponential':
        initial_learning_rate = .01
        decay_steps = 10000
        decay_rate = .96
    
        lr_schedule = KerasSchedules.ExponentialDecay(initial_learning_rate,
                                                      decay_steps=decay_steps,
                                                      decay_rate=decay_rate,
                                                      staircase=True)
        
    if decay == 'warmup':
        # Learning rate schedule
        LR_START = 0.00001
        LR_MAX = 0.00002 
        LR_MIN = 0.00001
        LR_RAMPUP_EPOCHS = 3
        LR_SUSTAIN_EPOCHS = 1
        LR_EXP_DECAY = (nepochs - (LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS)) / nepochs
        assert nepochs > 7 and isinstance(nepochs, int), \
            f"Number of epochs must be greater than 7, got: {nepochs}, You stack in rising lr"
            

        def lrfn(epoch):
            if epoch < LR_RAMPUP_EPOCHS:
                lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
            elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
                lr = LR_MAX
            else:
                lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
            return lr


        # lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True) # unsuported class in our approach
        
        from matplotlib import pyplot as plt
        rng = [i for i in range(nepochs)]
        y = [lrfn(x) for x in rng]
        plt.plot(rng, y)
        print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
        
        lr_schedule = KerasSchedules.PiecewiseConstantDecay(rng[:-1], y)

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

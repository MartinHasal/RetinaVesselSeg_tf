from enum import Enum
from tensorflow.keras.optimizers import schedules as KerasSchedules


class LearningRateDecayType(Enum):

    EXPONENTIAL = 'exponential'
    WARMUP_EXPONENTIAL_DECAY = 'warmup'

    def __str__(self):
        return self.value


def ScheduleWarmupExponentialDecay(nepochs: int, info: bool = False):

    LR_START = 0.00001
    LR_MAX = 0.00002
    LR_MIN = 0.00001
    LR_RAMPUP_EPOCHS = 3
    LR_SUSTAIN_EPOCHS = 1
    LR_EXP_DECAY = (nepochs - (LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS)) / nepochs

    def lrfn(epoch):
        if epoch < LR_RAMPUP_EPOCHS:
            lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
        elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
            lr = LR_MAX
        else:
            lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
        return lr

    assert nepochs > 7 and isinstance(nepochs, int), \
        f"Number of epochs must be greater than 7, got: {nepochs}, You stack in rising lr"

    rng = range(nepochs)
    y = [lrfn(x) for x in rng]

    if info:
        from matplotlib import pyplot as plt

        plt.plot(rng, y)
        print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))

    return KerasSchedules.PiecewiseConstantDecay(rng[:-1], y)

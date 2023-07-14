import argparse

from enum import Enum

from procs.adapter import DatasetAugmentation
from funcs.losses import LossType
from funcs.lr import LearningRateDecayType


class _DatasetAugmentationOps_Helper(Enum):

    NONE = 'none'
    CLAHE_EQUALIZATION = 'clahe_equalization'
    CLAHE_EQUALIZATION_INPLACE = 'clahe_equalization_inplace'
    FLIP_HORIZONTAL = 'flip_horizontal'
    FLIP_VERTICAL = 'flip_vertical'
    ROTATION_90 = 'rotation_90'
    ADJUST_CONTRAST = 'adjust_contrast'
    ADJUST_BRIGHTNESS = 'adjust_brightness'
    ALL = 'all'
    ALL_CLAHE_INPLACE = 'all_inplace'

    def __str__(self):

        return self.value


def process_augmentation_ops(ds_augmentation_ops) -> list:

    lst_ops = []

    for op in ds_augmentation_ops:

        if op == _DatasetAugmentationOps_Helper.NONE:
            lst_ops = [ds_augmentation_ops.NONE]
            return lst_ops

        if op == _DatasetAugmentationOps_Helper.CLAHE_EQUALIZATION:
            lst_ops.append(DatasetAugmentation.CLAHE_EQUALIZATION)

        if op == _DatasetAugmentationOps_Helper.CLAHE_EQUALIZATION_INPLACE:
            lst_ops.append(DatasetAugmentation.CLAHE_EQUALIZATION_INPLACE)

        if op == _DatasetAugmentationOps_Helper.FLIP_HORIZONTAL:
            lst_ops.append(DatasetAugmentation.FLIP_HORIZONTAL)

        if op == _DatasetAugmentationOps_Helper.FLIP_VERTICAL:
            lst_ops.append(DatasetAugmentation.FLIP_VERTICAL)

        if op == _DatasetAugmentationOps_Helper.ROTATION_90:
            lst_ops.append(DatasetAugmentation.ROTATION_90)

        if op == _DatasetAugmentationOps_Helper.ADJUST_CONTRAST:
            lst_ops.append(DatasetAugmentation.ADJUST_CONTRAST)

        if op == _DatasetAugmentationOps_Helper.ADJUST_BRIGHTNESS:
            lst_ops.append(DatasetAugmentation.ADJUST_BRIGHTNESS)

        if op == _DatasetAugmentationOps_Helper.ALL:
            lst_ops = [DatasetAugmentation.ALL]
            return lst_ops

        if op == _DatasetAugmentationOps_Helper.ALL_CLAHE_INPLACE:
            lst_ops = [DatasetAugmentation.ALL_CLAHE_INPLACE]
            return lst_ops

    return lst_ops


class Range(object):

    def __init__(self, start, end):
        self.start = start
        self.end = end
        
    def __eq__(self, other):
        return self.start <= other <= self.end
        
    def __repr__(self):
        return '{0} - {1}'.format(self.start, self.end)


def cli_argument_parser() -> dict:

    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--db_csv',
                        metavar='CSV_FILE',
                        type=str,
                        default='dataset.csv',
                        required=False)

    parser.add_argument('--output_model_path',
                        metavar='PATH',
                        required=False)

    parser.add_argument('--output_model_name',
                        metavar='MODEL_NAME',
                        default='unet_vgg16',
                        required=False)

    parser.add_argument('--patch_size',
                        metavar='PATCH_SIZE',
                        default=128,
                        required=False,
                        type=int)

    parser.add_argument('--patch_overlap_ratio',
                        metavar='PATCH_OVERLAP_RATIO',
                        default=.5,
                        required=False,
                        type=float,
                        choices=[Range(0.0, 1.0)])

    parser.add_argument('--ds_augmentation_ratio',
                        metavar='AUGMENTATION_RATIO',
                        default=.5,
                        required=False,
                        type=float,
                        choices=[Range(0.0, 1.0)])

    parser.add_argument('--clahe_augmentation_ratio',
                        metavar='CLAHE_AUGMENTATION_RATIO',
                        default=.1,
                        required=False,
                        type=float,
                        choices=[Range(0.0, 1.0)])

    parser.add_argument('--ds_test_ratio',
                        metavar='TEST_DATASET_RATIO',
                        default=.1,
                        required=False,
                        type=float,
                        choices=[Range(0.0, 1.0)])

    parser.add_argument('--batch_size',
                        metavar='BATCH_SIZE',
                        default=32,
                        required=False,
                        type=int)

    parser.add_argument('--nepochs',
                        metavar='NUMBER_OF_EPOCHS',
                        default=30,
                        required=False,
                        type=int)

    parser.add_argument('--loss_type',
                        metavar='LOSS_FUNCTION_TYPE',
                        type=LossType,
                        choices=LossType,
                        default=LossType.CROSS_ENTROPY,
                        required=False)

    parser.add_argument('--lr_decay_type',
                        metavar='LEARNING_RATE_DECAY_TYPE',
                        type=LearningRateDecayType,
                        choices=LearningRateDecayType,
                        default=LearningRateDecayType.WARMUP_EXPONENTIAL_DECAY,
                        required=False)

    parser.add_argument('--ds_augmentation_ops',
                        metavar='OP1 OP2 OP3',
                        nargs='+',
                        type=_DatasetAugmentationOps_Helper,
                        choices=_DatasetAugmentationOps_Helper,
                        required=False)

    parser.add_argument('--model_trainable_encoder',
                        metavar='True or False',
                        default=False,
                        required=False)
                        
    parser.add_argument('--crop_threshold',
                        metavar="[0-255]",
                        help='Threshold taken from range (0-255) denotes at what grayscale threshold level \
                        the margins from images are being cropped. Default is -1 (no crop).',
                        type=int,
                        choices=range(-1, 256),
                        default=-1,
                        required=False)

    args = parser.parse_args()
    lst_ops = process_augmentation_ops(args.ds_augmentation_ops) if args.ds_augmentation_ops is not None else (DatasetAugmentation.NONE,)

    kwargs = {
        'db_name': args.db_csv,
        'output_model_path': args.output_model_path,
        'output_model_name': args.output_model_name,
        'patch_size': args.patch_size,
        'patch_overlap_ratio': args.patch_overlap_ratio,
        'ds_augmentation_ratio': args.ds_augmentation_ratio,
        'ds_test_ratio': args.ds_test_ratio,
        'batch_size': args.batch_size,
        'nepochs': args.nepochs,
        'loss_type': args.loss_type,
        'lr_decay_type': args.lr_decay_type,
        'clahe_augmentation_ratio': args.clahe_augmentation_ratio,
        'ds_augmentation_ops': lst_ops,
        'trainable_encoder': args.model_trainable_encoder,
        'crop_threshold': args.crop_threshold
    }

    return kwargs

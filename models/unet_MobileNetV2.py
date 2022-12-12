import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import re

import tensorflow as tf

from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, Dropout, Input
from keras.engine.functional import Functional as KerasFunctional
from keras.models import Model as KerasModel







class UnetMobileNetV2(object):
    def __init__(self,
                 input_shape: tuple[int, int],
                 nclasses: int,
                 encoder_type: str = 'MobileNetV2',
                 trainable_encoder: bool = False):
        
        self._input_shape = input_shape
        self._nclasses = nclasses

        self._encoder_type = encoder_type
        self._trainable_encoder = trainable_encoder
        
        self._nn_model = None
        
    @property
    def nclasses(self) -> int:

        return self._nclasses

    @nclasses.setter
    def nclasses(self,
                 n: int):

        if self._nn_model is not None:
            del self._nn_model; gc.collect()
            self._nn_model = None

        self._nclasses = n

    @property
    def trainable_encoder(self):

        return self._trainable_encoder

    @trainable_encoder.setter
    def trainable_encoder(self,
                          flag: bool):

        if self._nn_model is not None:
            del self._nn_model; gc.collect()
            self._nn_model = None

        self._trainable_encoder = flag
    
    @property
    def model(self) -> KerasModel:

        if self._nn_model is None:
            self.build()

        return self._nn_model
    
    
    
    @staticmethod
    def upsample(filters, size, name):
      return tf.keras.Sequential([
             tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same'),
             tf.keras.layers.BatchNormalization(),
             tf.keras.layers.ReLU()
             ], name=name)

       
    def __buildUnet_MobileNetV2(self) -> KerasModel:
        
        base_model = tf.keras.applications.MobileNetV2(input_shape=self._input_shape,
                                                       include_top=False,
                                                       weights='imagenet')

        # Use the activations of these layers to form skip connections
        layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]
        
        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    
        # Create the feature extraction model
        down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs,
                                    name='pretrained_mobilenet')
    
        down_stack.trainable = self._trainable_encoder
        
        up_stack = [
            self.upsample(512, 3, 'upsample_4x4_to_8x8'),
            self.upsample(256, 3, 'upsample_8x8_to_16x16'),
            self.upsample(128, 3, 'upsample_16x16_to_32x32'),
            self.upsample(64, 3,  'upsample_32x32_to_64x64')
        ]
        
        inputs = tf.keras.layers.Input(shape=self._input_shape,
                                       name='input_image')
    
        # Downsampling through the model
        skips = down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])
    
        # Upsampling and establishing the skip connections
        for idx, (up, skip) in enumerate(zip(up_stack, skips)):
            x = up(x)
            concat = tf.keras.layers.Concatenate(name='expand_{}'.format(idx))
            x = concat([x, skip])
    
          # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
              self._nclasses, 3, strides=2,
              activation='softmax',
              padding='same')  #64x64 -> 128x128
    
        x = last(x)
        
        self._nn_model = tf.keras.Model(inputs=inputs, outputs=x, name='UNet_MobileNetV2')
        return self._nn_model


    
    
    def build(self) -> KerasModel:

        if self._encoder_type == 'MobileNetV2':
            self.__buildUnet_MobileNetV2()
        else:
            NotImplementedError('An UNet implementation just supports only MobileNetV2 encoder.')

        return self._nn_model








# tests
if __name__ == '__main__':

    unet = UnetMobileNetV2(
        input_shape=(128, 128, 3),
        nclasses=2,
        encoder_type='MobileNetV2'
    )

    nn_model = unet.model
    nn_model.summary()

    unet.trainable_encoder = True
    nn_model = unet.model
    nn_model.summary()

    unet.with_dropout = True
    nn_model = unet.model
    nn_model.summary()

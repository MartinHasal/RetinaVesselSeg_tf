import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, Dropout, Input
from keras.engine.functional import Functional as KerasFunctional
from keras.models import Model as KerasModel

from keras.applications.vgg16 import VGG16


class UNet(object):

    def __init__(self,
                 input_shape: tuple,
                 nclasses: int,
                 with_dropout: bool = False,
                 dropout_rate: float = 0.2,
                 encoder_type: str = 'vgg16',
                 trainable_encoder: bool = False
                 ) -> None:

        self._input_shape = input_shape
        self._nclasses = nclasses

        self._encoder_type = encoder_type
        self._trainable_encoder = trainable_encoder

        self._with_dropout = with_dropout
        self._dropout_rate = dropout_rate

        self._nn_model = None

    @staticmethod
    def encoder_vgg16(layer_input,
                      trainable: bool = False
                      ) -> KerasFunctional:

        # load vgg16 network (pretrained on ImageNet)
        # setting include_top to False discards fully connected layers at top of network
        # moreover, this reduces #parameters from 138M to 14.7M
        nn_vgg16 = VGG16(
            include_top=False,
            weights='imagenet',
            input_tensor=layer_input
        )

        # enable/disable for retraining encoder
        nn_vgg16.trainable = trainable

        return nn_vgg16

    @property
    def with_dropout(self) -> bool:

        return self._with_dropout

    @with_dropout.setter
    def with_dropout(self,
                     flag: bool):

        if self._nn_model is not None:
            del self._nn_model; gc.collect()
            self._nn_model = None

        self._with_dropout = flag

    @property
    def dropout_rate(self) -> float:

        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self,
                     rate: float):

        if self._nn_model is not None:
            del self._nn_model; gc.collect()
            self._nn_model = None

        self._dropout_rate = rate

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

    def conv_block(self,
                   input_layer: KerasFunctional,
                   nfilters: int
                   ) -> KerasFunctional:

        layer = Conv2D(
            nfilters,
            kernel_size=(3, 3),
            padding='same'
        )(input_layer)

        layer = Dropout(self._dropout_rate)(layer) if self._with_dropout else BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        return layer

    def decoder_block(self,
                      input_layer: KerasFunctional,
                      skip_connections: KerasFunctional,
                      nfilters: int,
                      kernel_size: tuple,
                      ) -> KerasFunctional:

        upsampling_layer = Conv2DTranspose(
            nfilters,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding='same',
            dtype='float32'
        )(input_layer)

        # Concatenated skip connections - reusing features from previous filters (convolution layer) by concatenating
        # to decoder layer. This allows retaining more information from previous filter which can lead to better
        # gradient propagation (preventing gradient vanishing) across the network and to achieve a better performance.
        layer = Concatenate()([upsampling_layer, skip_connections])

        layer = self.conv_block(
            layer,
            nfilters
        )
        layer = self.conv_block(
            layer,
            nfilters
        )

        return layer

    def __buildUnet_EncoderVGG16(self) -> KerasModel:

        # declare input layer
        layer_input = Input(self._input_shape)

        # get pretrained vgg16 encoder
        nn_vgg16 = self.encoder_vgg16(layer_input, self._trainable_encoder)

        # encoder: Get skip connections
        skip_layer_1 = nn_vgg16.get_layer('block1_conv2')
        skip_layer_2 = nn_vgg16.get_layer('block2_conv2')
        skip_layer_3 = nn_vgg16.get_layer('block3_conv3')
        skip_layer_4 = nn_vgg16.get_layer('block4_conv3')

        # bridge between encoder and decoder
        bridge_layer_1 = nn_vgg16.get_layer('block5_conv3').output

        # decoder: define architecture
        decoder_block_1 = self.decoder_block(
            input_layer=bridge_layer_1,
            skip_connections=skip_layer_4.output,
            nfilters=skip_layer_4.output.shape[3],
            kernel_size=skip_layer_4.get_config()['kernel_size']
        )

        decoder_block_2 = self.decoder_block(
            input_layer=decoder_block_1,
            skip_connections=skip_layer_3.output,
            nfilters=skip_layer_3.output.shape[3],
            kernel_size=skip_layer_3.get_config()['kernel_size']
        )

        decoder_block_3 = self.decoder_block(
            input_layer=decoder_block_2,
            skip_connections=skip_layer_2.output,
            nfilters=skip_layer_2.output.shape[3],
            kernel_size=skip_layer_2.get_config()['kernel_size']
        )

        decoder_block_4 = self.decoder_block(
            input_layer=decoder_block_3,
            skip_connections=skip_layer_1.output,
            nfilters=skip_layer_1.output.shape[3],
            kernel_size=skip_layer_2.get_config()['kernel_size']
        )

        # classification layer
        layer_output = Conv2D(
            self._nclasses, 1,
            padding='same',
            activation='softmax',
            dtype='float32'
        )(decoder_block_4)

        # declare neural network model
        self._nn_model = KerasModel(
            inputs=layer_input,
            outputs=layer_output,
            name='UNet_VGG16Encoder'
        )

        return self._nn_model

    def build(self) -> KerasModel:

        if self._encoder_type.lower() == 'vgg16':
            self.__buildUnet_EncoderVGG16()
        else:
            NotImplementedError('An UNet implementation just supports only vgg16 encoder.')

        return self._nn_model


# tests
if __name__ == '__main__':

    unet = UNet(
        input_shape=(128, 128, 3),
        nclasses=10,
        encoder_type='vgg16'
    )

    nn_model = unet.model
    nn_model.summary()

    unet.trainable_encoder = True
    nn_model = unet.model
    nn_model.summary()

    unet.with_dropout = True
    nn_model = unet.model
    nn_model.summary()

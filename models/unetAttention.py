import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers import Input, Activation, BatchNormalization, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, UpSampling2D, Dropout
from keras.engine.functional import Functional as KerasFunctional
from keras.models import Model as KerasModel, Model
from tensorflow.keras import Input
from tensorflow.keras.layers import multiply, add

from keras.applications.vgg16 import VGG16



class AttentionUNet(object):
    def __init__(self,
                 input_shape: tuple,
                 nclasses: int,
                 encoder_type: str = 'attentionunet',
                 ) -> None:

        self._input_shape = input_shape
        self._nclasses = nclasses
        self._encoder_type = encoder_type
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
    def model(self) -> KerasModel:

        if self._nn_model is None:
            self.build()

        return self._nn_model
        
        
    def conv_block(self, input_layer: KerasFunctional, filters: int):
        
        x = Conv2D(filters,(3,3),padding='same',activation='relu')(input_layer)
        x = Conv2D(filters,(3,3),padding='same')(x)
        x = BatchNormalization(axis=3)(x)
        # here think about dropout layer
        x = Activation('relu')(x)         
        return x


    def encoder_block(self, inp: KerasFunctional, filters: int):
        
        x = self.conv_block(inp,filters)
        p = MaxPooling2D(pool_size=(2,2))(x)        
        return x, p

    # attention block
    def attention_block(self, l_layer: KerasFunctional, h_layer: KerasFunctional): 
        phi = Conv2D(h_layer.shape[-1],(1,1),padding='same')(l_layer)
        theta = Conv2D(h_layer.shape[-1],(1,1),strides=(2,2),padding='same')(h_layer)
        x = add([phi,theta])
        x = Activation('relu')(x)
        x = Conv2D(1,(1,1),padding='same',activation='sigmoid')(x)
        x = UpSampling2D(size=(2,2))(x)
        x = multiply([h_layer,x])
        x = BatchNormalization(axis=3)(x)
        return x

        
    def decoder_block(self, input_layer: KerasFunctional, filters: int, concat_layer: KerasFunctional):
        x = Conv2DTranspose(filters,(2,2),strides=(2,2),padding='same')(input_layer)
        concat_layer = self.attention_block(input_layer, concat_layer)
        x = concatenate([x,concat_layer])
        x = self.conv_block(x,filters)
        return x    
    
    

    def __buildUnet_EncoderVGG16(self) -> KerasModel:
        
        inputs=Input(self._input_shape)
        d1, p1 = self.encoder_block(inputs, 64)
        d2, p2 = self.encoder_block(p1, 128)
        d3, p3 = self.encoder_block(p2, 256)
        d4, p4 = self.encoder_block(p3, 512)
        b1 = self.conv_block(p4, 1024)
        e2 = self.decoder_block(b1, 512, d4)
        e3 = self.decoder_block(e2, 256, d3)
        e4 = self.decoder_block(e3, 128, d2)
        e5 = self.decoder_block(e4 ,64, d1)
        outputs = Conv2D(self._nclasses, (1,1), activation="softmax", dtype='float32')(e5)
        
        # declare neural network model
        self._nn_model = Model(
            inputs=inputs,
            outputs=outputs,
            name='AttentionUNet'
        )

        return self._nn_model

    def build(self) -> KerasModel:

        if self._encoder_type.lower() == 'attentionunet':
            self.__buildUnet_EncoderVGG16()
        else:
            NotImplementedError('An AttentionUNet implementation is not right.')

        return self._nn_model


    


# tests
if __name__ == '__main__':

    unet = AttentionUNet(
        input_shape=(256, 256, 3),
        nclasses=1,
        encoder_type='attentionunet'
    )

    nn_model = unet.model
    nn_model.summary()

    # unet.trainable_encoder = True
    # nn_model = unet.model
    # nn_model.summary()

    # unet.with_dropout = True
    # nn_model = unet.model
    # nn_model.summary()

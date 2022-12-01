# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:02:58 2022

@author: has081
U-net segmentace

First, basically naive version of U-net segmentation

"""
# imports
import pandas as pd
import cv2
import os, shutil
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import argparse
from sklearn.model_selection import train_test_split
#import apache_beam as beam
import tensorflow as tf


IMAGE_SIZE = (128, 128) 
IMG_CHANNELS = 3


# read the dataset
def read_df(location, col_names=None):
    if col_names:
        return pd.read_csv(location, names=col_names, index_col=['index'])
    else:
        return pd.read_csv(location,index_col=['index'])
    





def read_and_decode_RGB_cv2(filename):        
    img = cv2.imread(filename,cv2.IMREAD_UNCHANGED) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE) 
    
    return img



def read_and_decode_mask_cv2(filename):    
    # gif cannot be read by opencv
    if filename.split('.')[-1] == 'gif':
        pil_img = Image.open(filename)
        img = np.array(pil_img)
    else:
        img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)          
    
    img  = cv2.resize(img, IMAGE_SIZE, interpolation = cv2.INTER_NEAREST)
    return img


def decode_name(line):

    img_path, mask_path = line[0].numpy(), line[1].numpy()
    img = read_and_decode_RGB_cv2(img_path)
    mask = read_and_decode_mask_cv2(mask_path)
   
    return img, mask



def get_images(df, col_names=['PATH_TO_ORIGINAL_IMAGE', 'MASK']):
    df = df[col_names]
    #img = np.empty([len(df), 128,128,3])
    #mask = np.empty([len(df), 128,128,3])
    img = []
    mask = []
    
    # i=0
    for index, row in df.iterrows():
        # print(row[col_names[0]], row[col_names[1]])
        #img[i,:,:,:] = read_and_decode_RGB_cv2(row[col_names[0]])
        #mask[i,:,:,:] = read_and_decode_mask_cv2(row[col_names[1]])
        img.append( read_and_decode_RGB_cv2(row[col_names[0]]) )
        mask.append(read_and_decode_mask_cv2(row[col_names[1]]) )
        # print(img_shape, mask_shape)
        
    return img, mask


if __name__ == "__main__":
    # space for argpars
    
    
    PATH = os.getcwd()
    DATABASE_CSV_NAME = 'data_paths.csv'
    COLUMNS = ['PATH_TO_ORIGINAL_IMAGE', 'MASK']
    
    
    
    df = read_df(os.path.join(PATH, DATABASE_CSV_NAME))
    train, valid = train_test_split(df, test_size=0.2,random_state=42)
    
    train_img, mask_img =  get_images(train, col_names=['PATH_TO_ORIGINAL_IMAGE', 'MASK'])
    mask_img = np.expand_dims(mask_img, axis=3) # not be necessary in some cases
    # pretrained version
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000

    # does nto work
    # using a `tf.Tensor` as a Python `bool` is not allowed in Graph execution. Use Eager execution or decorate this function with @tf.function.
    # condition is problem
    def augment_old(img, mask):
      if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
      return img, mask



    def augment(img, mask):    
        return tf.cond(pred= tf.random.uniform(()) > 0.5,
                       true_fn=(lambda  : (tf.image.flip_left_right(img), tf.image.flip_left_right(mask))),
                       false_fn=(lambda   : (img, mask))  
            )
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_img,mask_img))
    train_dataset = train_dataset.cache().map(augment).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    
    valid_img, valid_mask =  get_images(train, col_names=['PATH_TO_ORIGINAL_IMAGE', 'MASK'])
    valid_mask = np.expand_dims(valid_mask, axis=3) # not be necessary in some cases
    
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_img,valid_mask)).batch(BATCH_SIZE)
    
    
    TRAIN_LENGTH = len(train)
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    OUTPUT_CHANNELS = 1

    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

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

    down_stack.trainable = False

    def upsample(filters, size, name):
      return tf.keras.Sequential([
         tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same'),
         tf.keras.layers.BatchNormalization(),
         tf.keras.layers.ReLU()
      ], name=name)

    up_stack = [
        upsample(512, 3, 'upsample_4x4_to_8x8'),
        upsample(256, 3, 'upsample_8x8_to_16x16'),
        upsample(128, 3, 'upsample_16x16_to_32x32'),
        upsample(64, 3,  'upsample_32x32_to_64x64')
    ]

    import re
    
    def unet_model(output_channels):
      inputs = tf.keras.layers.Input(shape=[128, 128, 3], name='input_image')

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
          output_channels, 3, strides=2,
          padding='same')  #64x64 -> 128x128

      x = last(x)

      return tf.keras.Model(inputs=inputs, outputs=x)

    model = unet_model(OUTPUT_CHANNELS)
    
    model = unet_model(OUTPUT_CHANNELS)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['mse'])


    # Assign to the pixel the label with the highest probability
    def create_mask(pred_mask):
      pred_mask = tf.argmax(pred_mask, axis=-1)
      pred_mask = pred_mask[..., tf.newaxis]
      return pred_mask[0]

    # display helper functions
    def display(display_list):
      plt.figure(figsize=(15, 15))

      title = ['Input Image', 'True Mask', 'Predicted Mask']

      for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        if i > 1:
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap='gray')
        else:
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
      plt.show()

    def show_predictions(dataset, num):
      for value, (image, mask) in enumerate(dataset.take(num)):
          if value == 6:
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
        
    # before the model is trained
    show_predictions(train_dataset, 7)

    # from IPython.display import clear_output
    class DisplayCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs=None):
        if epoch%50 == 0:
          # clear_output(wait=True) # if you want replace the images each time, uncomment this
          show_predictions(train_dataset, 7)
          print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


    EPOCHS = 200
    VAL_SUBSPLITS = 5
    VALIDATION_STEPS = len(valid)//BATCH_SIZE//VAL_SUBSPLITS

    model_history = model.fit(train_dataset, epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=valid_dataset,
                              callbacks=[DisplayCallback()])


    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']


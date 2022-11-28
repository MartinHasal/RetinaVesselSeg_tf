# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 09:07:19 2022

@author: has081 Martin Hasal

Due to various files of images and mask, and for effeciency, 
the TF Recods are used.

This program split a dataset into training, validation, testing 
and write those images into TensorFlow Record files

NOTE database structure is 
`NAME, DATASET_NAME, PATH_TO_ORIGINAL_IMAGE, MASK`

Important remove folder output and create a new one!!!
rm -rf output
mkdir output


ToDo:
    - DONE: big potential issue can be reading .ppm in read_and_decode_RGB
        because tf has no reader of ppm, hence it is bypassed by cv2
    - to previous point, keras ImageDataGenerator.read_from_files
        can read all types of files, but creation of tfrecods is not straightforward
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
import tensorflow.io as tfio



# read the dataset
def read_df(location, col_names=None):
    if col_names:
        return pd.read_csv(location, names=col_names, index_col=['index'])
    else:
        return pd.read_csv(location,index_col=['index'])

def get_dataset_extensions(df, col_names=['PATH_TO_ORIGINAL_IMAGE', 'MASK']):
    """ Helping fuction to get extension of files """
    extensions = []
    # iterate over columns and get extensions
    for col_indx in range(len(col_names)):
        column = df[col_names[col_indx]]
        # iterate over rows and find files extensions
        for row in column:
            # split by dot and get the last field
            extensions.append(row.split('.')[-1])
            
    return extensions    
    


# read and decode original image in different formats
# basically np.max(read_and_decode_RGB(JPG).numpy()) = 1
# but for np.max(read_and_decode_RGB(STARE).numpy()) = 255.0
# tf.image.convert_image_dtype does not work with convert_to_tensor
def read_and_decode_RGB(filename):
    IMG_CHANNELS = 3
    img = tf.io.read_file(filename)    
    extension = filename.split('.')[-1]
    if extension == 'jpg':
        img = tf.image.decode_jpeg(img, channels=IMG_CHANNELS)
    if extension == 'tif':
        img = tfio.experimental.image.decode_tiff(img, dtype=tf.float32)
        img = tfio.experimental.color.rgba_to_rgb(img)
    if extension == 'ppm':
        img = cv2.imread(filename,cv2.IMREAD_UNCHANGED) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tf.convert_to_tensor(img, dtype=tf.float32) # unit8 is not supported  
    
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


"""
# help code:
df_loc = df[df['NAME'] == '32_training']    
TIF = df_loc.PATH_TO_ORIGINAL_IMAGE.unique()[0]
read_and_decode_RGB_cv2(TIF)


df_loc = df[df['NAME'] == 'Image_01R']    
JPG = df_loc.PATH_TO_ORIGINAL_IMAGE.unique()[0]
read_and_decode_RGB_cv2(JPG)


df_loc = df[df['NAME'] == 'im0005']    
PPM = df_loc.PATH_TO_ORIGINAL_IMAGE.unique()[0]
read_and_decode_RGB_cv2(PPM)
"""

# read all images by cv2, slow, but there is an issue, see read_and_decode_RGB
# goal is not to do a mistake and have robust solution
def read_and_decode_RGB_cv2(filename):
        
    img = cv2.imread(filename,cv2.IMREAD_UNCHANGED) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.convert_to_tensor(img, dtype=tf.uint8)  
    
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def show_image(filename):
  img = read_and_decode_RGB_cv2(filename)
  plt.figure()
  plt.imshow( img.numpy() )
  plt.show()

"""
# help code:
df_loc = df[df['NAME'] == '32_training']    
GIF_MASK = df_loc.MASK.unique()[0]
read_and_decode_RGB_cv2(GIF_MASK)


df_loc = df[df['NAME'] == 'Image_01R']    
PNG_MASK = df_loc.MASK.unique()[0]
read_and_decode_RGB_cv2(PNG_MASK)


df_loc = df[df['NAME'] == 'im0005']    
PPM_MASK = df_loc.MASK.unique()[0]
read_and_decode_RGB_cv2(PPM_MASK)
"""


def read_and_decode_mask_cv2(filename):
    
    # gif cannot be read by opencv
    if filename.split('.')[-1] == 'gif':
        pil_img = Image.open(filename)
        img = np.array(pil_img)
    else:
        img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE) 
        img = tf.convert_to_tensor(img, dtype=tf.uint8)  
    
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def show_mask(filename):
  img = read_and_decode_mask_cv2(filename)
  plt.figure()
  plt.imshow( img.numpy(), cmap='gray', vmin=0, vmax=1)
  plt.show()


def plot_orig_mask(df, name):
    # function to print original image with masks
    df_loc = df[df['NAME'] == name]
    orig_img = cv2.imread(df_loc.PATH_TO_ORIGINAL_IMAGE.unique()[0],cv2.IMREAD_UNCHANGED)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # number of figures depends on number of masks
    fig = plt.figure(figsize=( 6*len(df_loc), 6))
    fig.subplots_adjust()
    for i in range(len(df_loc) + 1):
        ax = fig.add_subplot(1, 3, i + 1)
        if i == 0: # print ortiginal image
            ax.imshow(orig_img)
            ax.title.set_text(name)
        else:
            # masks
            path = df_loc.MASK.iloc[i-1]
            # check .gif due to DRIVE database
            if path[-3:] == 'gif':
                im = Image.open(path)
                mask_img = np.array(im)
            else:            # readable by OpenCV
                mask_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            ax.imshow(mask_img, cmap='gray')
            ax.title.set_text(path.split('\\')[-1])
    fig = ax.get_figure()
    fig.tight_layout()  


############ TF RECORDS

def _string_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_tfrecord(filename, label, label_int):
    print(filename)
    img = read_and_decode_RGB_cv2(filename)
    dims = img.shape
    img = tf.reshape(img, [-1]) # flatten to 1D array
    
    
    return tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _float_feature(img),
        'shape_image_raw': _int64_feature([dims[0], dims[1], dims[2]]),
        'mask_raw': _string_feature(label),
        'shape_mask_raw': _int64_feature([label_int])
    })).SerializeToString()

    

if __name__ == "__main__":
    # space for argpars
    
    
    PATH = os.getcwd()
    DATABASE_CSV_NAME = 'data_paths.csv'
    COLUMNS = ['PATH_TO_ORIGINAL_IMAGE', 'MASK']
    
    df = read_df(os.path.join(PATH, DATABASE_CSV_NAME))
    train, valid = train_test_split(df, test_size=0.2,random_state=42)
    
    extension = get_dataset_extensions(df, COLUMNS)
    #These are the files, that need to loaded by tf
    print(f'Unique extensions are {set(extension)}')
    
    dir = 'output'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    
    plot_orig_mask(df, df.NAME.iloc[110])
    
    train[COLUMNS].to_csv('output/train.csv', header=False, index=False)
    valid[COLUMNS].to_csv('output/valid.csv', header=False, index=False)

    


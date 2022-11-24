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
"""

# imports 
import pandas as pd
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import argparse



# read the dataset
def read_df(location, col_names=None):
    if col_names:
        return pd.read_csv(location, names=col_names)
    else:
        return pd.read_csv(location)
    

if __name__ == "__main__":
    # space for argpars
    
    
    PATH = os.getcwd()
    DATABASE_CSV_NAME = 'data_paths.csv'
    
    df = read_df(os.path.join(PATH, DATABASE_CSV_NAME))



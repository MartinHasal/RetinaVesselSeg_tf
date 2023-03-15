# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:36:51 2022

@author: has081


Read and decode datasets
Three datasets are available CHASEDB1, DRIVE, and STARE


All images will be saved in dataframe with structure
NAME, DATASET_NAME, PATH_TO_ORIGINAL_IMAGE, MASK

mulptiple masks will be stored again under same name
"""


# imports
import cv2
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import argparse


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    # save images from matplotlib
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def plot_fig(path):
    colormap1 = cv2.imread(path)

    img = cv2.cvtColor(colormap1, cv2.COLOR_BGR2RGB)
    # save_fig(name)
    plt.figure()
    plt.imshow(img)
    plt.title(path.split('\\')[-1])


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


def path_replace_HFR(line):
    # change file extension
    new_line = line[:-4]
    new_line = new_line + '.tif'
    path, base = os.path.split(new_line)
    parent_path = os.path.dirname(path)  
    return os.path.join(parent_path,'manual1',base)

""" 
os functions
"""


def get_files(path, endswith=".jpg"):
    # load images
    files_original = []
    images_stack = []
    images_folders = []

    # find only .jpg 
    for r, d, f in os.walk(path):
        # sorting files alphabetically for Unix-like systems
        for file in sorted(f):
            if file.endswith(endswith):
                #print(os.path.join(r, file))
                files_original.append(os.path.join(r, file))
                # stack names
                images_stack.append(file)
        images_folders.append(r)
    
    return files_original, images_stack, images_folders
    
 
# print the path to dataset
class VerboseStoreDataset(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError('nargs not allowed')
        super(VerboseStoreDataset, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print('Dataset saving, name %r defined by user' % (values))
        setattr(namespace, self.dest, values)    
    

#################################
if __name__ == "__main__":
    
    prog_desc = "Load data from datasets folder \n and creates csv with path to image and label"

    parser = argparse.ArgumentParser(description=prog_desc)
    parser.version = '1.2'
    # Add the arguments
    parser.add_argument('--path',
                        #metavar='path',
                        action='store',
                        type=str,
                        default=os.getcwd(),
                        help='the path to dataset, datased included. Default=current dir',
                        required=False)
    
    parser.add_argument('-b',
                        '--save_hrf',
                        action='store_true',
                        help='include HFR dataset 3504 x 2336.csv')
    
    parser.add_argument('-s',
                        '--save',
                        action='store_true',
                        help='store the dataset in data_paths.csv')
    
    parser.add_argument('-c',
                        '--csv',
                        type=str,
                        action=VerboseStoreDataset,
                        help='store dataset in user defined *name*.csv ')
    
    args = parser.parse_args()
    
    
    # set path to datasets
    # PATHCWD = 'D:\\ARG@CS.FEI.VSB Dropbox\\Martin Hasal\\Dataset - retiny\\images_from_doctor\\ML\\SEGMENTATION\\VESSELS\\datasets'
    PATHCWD = os.path.join(args.path, 'datasets')
    PATH_CHASEDB1 = os.path.join(PATHCWD, 'CHASEDB1')
    PATH_DRIVE = os.path.join(PATHCWD, 'DRIVE')
    PATH_STARE = os.path.join(PATHCWD, 'STARE', 'vessels')
    PATH_HRF = os.path.join(PATHCWD, 'HRF')

    """ Images functions """

    # Where to save the figures
    PROJECT_ROOT_DIR = "."
    CHAPTER_ID = "saved_images"
    IMAGES_PATH = os.path.join(PATHCWD,  CHAPTER_ID)
    os.makedirs(IMAGES_PATH, exist_ok=True)
    
    """ 
    
    CHASEDB1 DATASASET  --- all images in single folder
    
    """ 
    paths, names, _ = get_files(PATH_CHASEDB1)

    ##### important model, here the df is created
    df_chase = pd.DataFrame({ 
                      'NAME': [name[:-4] for name in names] ,
                      'DATASET_NAME': ['CHASEDB1']*len(names),
                      'PATH_TO_ORIGINAL_IMAGE': paths
                      })
    
    # because there are two mask for every image, df is doubled
    df_chase = pd.DataFrame(np.repeat(df_chase.values, 2, axis=0), columns=df_chase.columns)
    
    # get masks, in .png
    paths, names, _ = get_files(PATH_CHASEDB1, endswith=".png")
    df_chase['MASK'] = paths

    df = df_chase

    #for i in range(10):
    #    plot_fig(df.iloc[i,2])
    
    #plot_orig_mask(df, df.NAME.iloc[10])
    
    
    """ 
    
    CHASEDB1 DRIVE  --- all images in single folder
    https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction
    """ 
    paths, names, _ = get_files(os.path.join(PATH_DRIVE, 'training'), endswith=".tif")
    
    df_drive = pd.DataFrame({ 
                      'NAME': [name[:-4] for name in names] ,
                      'DATASET_NAME': ['DRIVE']*len(names),
                      'PATH_TO_ORIGINAL_IMAGE': paths
                      })
    
    paths, names, _ = get_files(os.path.join(PATH_DRIVE,'training','1st_manual'), endswith=".gif")
    df_drive['MASK'] = paths
    
    
    # concatenate
    df = pd.concat([df, df_drive])
    df_drive
    # plot_fig(df.iloc[70,3]), problem to read .gif
    #plot_orig_mask(df, df.NAME.iloc[70])
    
    
    """ 
    
    STARE  --- .ppm images
    
    """ 
    
    paths, names, _ = get_files(os.path.join(PATH_STARE,'stare-images'), endswith=".ppm")
    
    # for path in paths[:20]:
    #     img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     plt.figure()
    #     plt.imshow(img)
    #     plt.show()
        
    df_stare1 = pd.DataFrame({ 
                      'NAME': [name[:-4] for name in names],
                      'DATASET_NAME': ['STARE']*len(names),
                      'PATH_TO_ORIGINAL_IMAGE': paths
                      })
    
    paths, names, _ = get_files(os.path.join(PATH_STARE,'labels-ah'), endswith=".ppm")
    df_stare1['MASK'] = paths
    
    paths, names, _ = get_files(os.path.join(PATH_STARE,'stare-images'), endswith=".ppm")
    df_stare2 = pd.DataFrame({ 
                      'NAME': [name[:-4] for name in names],
                      'DATASET_NAME': ['STARE']*len(names),
                      'PATH_TO_ORIGINAL_IMAGE': paths
                      })
    
    paths, names, _ = get_files(os.path.join(PATH_STARE, 'labels-vk'), endswith=".ppm")
    df_stare2['MASK'] = paths
    
    df_stare = pd.concat([df_stare1, df_stare2])
    
    del df_stare1 
    del df_stare2
    df = pd.concat([df, df_stare])
    
    


    """ 
    
    HRF  --- .ppm images
    
    """ 
    if (args.save_hrf):
        paths, names, _ = get_files(os.path.join(PATH_HRF,'images'), endswith=".JPG")
        
        df_hfr_JPG = pd.DataFrame({ 
                          'NAME': [name[:-4] for name in names],
                          'DATASET_NAME': ['HFR']*len(names),
                          'PATH_TO_ORIGINAL_IMAGE': paths,
                          'MASK' : paths
                          })
        
        paths, names, _ = get_files(os.path.join(PATH_HRF,'images'), endswith=".jpg")
        
        df_hfr_jpg = pd.DataFrame({ 
                          'NAME': [name[:-4] for name in names],
                          'DATASET_NAME': ['HFR']*len(names),
                          'PATH_TO_ORIGINAL_IMAGE': paths,
                          'MASK' : paths
                          })
        
        df_hfr = pd.concat([df_hfr_JPG, df_hfr_jpg])
        
        # set right path to mask
        df_hfr['MASK'] = df_hfr.MASK.apply(path_replace_HFR)
        
        del df_hfr_JPG 
        del df_hfr_jpg 
        
        df = pd.concat([df, df_hfr])
    
    # SAVE PATHS
    #plot_orig_mask(df, df.NAME.iloc[110])
    
    print(df.DATASET_NAME.value_counts())
    
    # save paths
    if (args.csv or args.save):
        if args.csv:
            df.reset_index().to_csv(args.csv, index=False)
        else: 
            df.reset_index().to_csv('data_paths.csv', index=False)

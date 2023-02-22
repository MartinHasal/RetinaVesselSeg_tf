


# RetinaVesselSeg_tf: Retinal/Fundus vessel segmentation by TensorFlow

This repository contains Python/OpenCV/Tensorflow implementation of vessel segmentation from retinal images. It is based on U-net architecture with a VGG-16 backbone (some variations are also implemented). The neural network predicts the probability for every pixel to be a vessel or not. These probabilities are further processed.

Segmentation is done on the patched (tiled) version of the original image with variable overlap. Patched images can be augmented and added to the training dataset. Some augmentation techniques are applied on the mask (rotation, flip, etc.), but some are not (contrast, brightness). 

After training, the image blending technique can be used to produce more smooth results. Different postprocessing visualization techniques are used to produce aesthetically pleasing or error-correction visualizations.

## Segmenatation is based on three datasets:

- [DRIVE](http://www.isi.uu.nl/Research/Databases/DRIVE/),
- [STARE](http://www.ces.clemson.edu/ahoover/stare/),
- [CHASE_DB1](https://blogs.kingston.ac.uk/retinal/chasedb1/) 
- [HRF](https://www5.cs.fau.de/research/data/fundus-images/) 
 
Note, **datasets are in `.gitignore`** use the following link to get data: [datasets](https://www.dropbox.com/sh/ck5pqz8c11whthn/AADeiK1aDIGdN9SPGr0o0msha?dl=0).
After that, add datasets to root folder `RetinaVesselSeg_tf`

## Basic workflow
1. Clone main branch
2. Download the data to `.\datasets` folder
3. run command  `python .\data_processing.py -s` in main folder. It creates `data_paths.csv` file with structure `NAME, DATASET_NAME, PATH_TO_ORIGINAL_IMAGE, MASK`  
Alternative name of csv,e.g., *data.csv*, is also possible by  `python .\data_processing.py -c 'data.csv'`.
By default `data_paths.csv` does not contain HRF dataset (3504 x 2336, creates huge dataset in pathed algorithm) to paths. It can be add by `python .\data_processing.py -b` argument.
4. In the folder run command `pipelines/pipeline_unet_vgg16.py --db_csv data_paths.csv`

# Advanced description
## Dataset creation
As was mentioned at the beginning, the training dataset is created from  DRIVE, STARE, and CHASEDB1 datasets. Possibly by change of argument `-b` in `python .\data_processing.py` the high-level-resolution HFR dataset can be added. The segmentation algorithm is based on patching and it adds significant amount of training images, and since the model cannot be trained on standard laptops (tested on Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz, 2592 Mhz, 6 Core(s), 12 Logical Processor(s), 32 GB RAM, NVIDIA GeForce RTX 2060 with Max-Q Design 6GB)

Python's skript `\data_processing.py` authomatically reads the structure of all datasets as they were originally created (some unusefull parts of datasets were manually removed in provided link). The outcome is `data_paths.csv` file with structure `NAME, DATASET_NAME, PATH_TO_ORIGINAL_IMAGE, MASK`. Alternative name of csv, is also possible by  `python .\data_processing.py -c 'data.csv'`.

Machine learning pipeline loads the dataset, reads the `PATH_TO_ORIGINAL_IMAGE` and `MASK`, loads the images, and does its job. Hence, this segmentation algorithm can be used for any task!!!

### Dataset crop
We observed that dataset contains images with black edges around retinal images and such information is redundant and not necessary. Patching creates many black patches with no information for segmentation algorithm. Consequently, we decided to crop such images to the retinal boundary. It can be done 
1. online during dataset loading by argument `--crop_val 21` in `pipelines/pipeline_unet_vgg16.py --db_csv data_paths.csv`. The value 21 is the level of gray in grayscale image, which is considered as black and edges with lower value are removed.
2. by script `data_processing_crop_images.py` which works as previously described `data_processing.py`, except it creates a new folder called *dataset_cropped*. It contains all cropped images stacked on in one folder. Also `data_paths_cropped.csv` is created with new paths pointing to this folder. Alternatively, user can check right value of ` data_processing_crop_images.py -crop_val` in the *dataset_cropped*. Or by setting it to 0, user can see the whole dataset in one place.

![alt text-1](https://github.com/MartinHasal/RetinaVesselSeg_tf/blob/main/readme_img/cropped.png?raw=true)![alt text-2](https://github.com/MartinHasal/RetinaVesselSeg_tf/blob/main/readme_img/cropped_80.png?raw=true)


# Run

The program runs automatically, by running the ML pipeline from *pipelines* folder. A few selfexpaining arguments can be changed. The main pipeline containig all visualizations is 
`pipelines/pipeline_unet_vgg16.py` 
with the following arguments
`python <scripts in pipelines>.py [--db_csv CSV_FILE] [--output_model_path PATH] [--output_model_name MODEL_NAME] [--patch_size PATCH_SIZE] [--patch_overlap_ratio PATCH_OVERLAP_RATIO] [--ds_augmentation_ratio AUGMENTATION_RATIO] [--clahe_augmentation_ratio CLAHE_AUGMENTATION_RATIO] [--ds_test_ratio TEST_DATASET_RATIO] [--batch_size BATCH_SIZE] [--nepochs NUMBER_OF_EPOCHS] [--loss_type LOSS_FUNCTION_TYPE] [--lr_decay_type LEARNING_RATE_DECAY_TYPE] [--ds_augmentation_ops OP1 OP2 OP3] [--model_trainable_encoder True or False] [--crop_val [0-255]]`

Arguments (N: stands for note)

    --db_csv: (optional) path to the input CSV file containing image and label paths (default: 'dataset.csv') N: pipepline can be used for any binary segmentation problem, just load paths to data in given format
    --output_model_path: (optional) path to save the trained model (default: None)
    --output_model_name: (optional) name of the trained model (default: 'unet_vgg16')
    --patch_size: (optional) size of image patches to be extracted (default: 128)
    --patch_overlap_ratio: (optional) overlap ratio of adjacent patches (default: 0.5)
    --ds_augmentation_ratio: (optional) ratio of augmented images to be added to the training set (default: 0.5) N: augmented images are stacked with original images
    --clahe_augmentation_ratio: (optional) ratio of CLAHE-augmented images to be added to the training set (default: 0.1) N: CLAHE method can be applied on image before patching
    --ds_test_ratio: (optional) ratio of images to be used for testing (default: 0.1)
    --batch_size: (optional) batch size for training (default: 32)
    --nepochs: (optional) number of epochs for training (default: 30)
    --loss_type: (optional) type of loss function to use (default: 'cross_entropy') N: others loss functions are implemented, but effect on accurancy is low
    --lr_decay_type: (optional) type of learning rate decay to use (default: 'warmup_exponential_decay') N: warm-up lr requires at least 7 epochs, for testing use 'exponential'
    --ds_augmentation_ops: (optional) list of dataset augmentation operations to apply (default: 'none') N: check to code for available augmentation
    --model_trainable_encoder: (optional) whether or not to make the encoder portion of the model trainable (default: False) N: keep in mind, it almost doubles the size of parameters
    --crop_val: (optional) threshold (0-255) denoting at what threshold of grayscale the black edges from images are cropped (default: 0) N: crop decreases the amount of traing images by approx. 10%, i.e., every epoch is trained faster. Black images are still part of patches do not worry :]s
    
## Augmentations
All augmentations techniques are implemented in TensorFlow but in our `class DataAdapter`, which also loads data and prepares train, validation, and test datasets.

## Visualizations

The final product of segmentation algorithm is the image of segmented blood vessels. Various methods were implemented to visualise results from image segmentation.


### Training history

### ROC curve

### Comparable results
The following images compare masks segmented by experts against the segmentation algorithm with probability value and label value. The motivation for the usage of probability value is twofold:
1. Dataset contains different masks for one retinal image, i.e. even experts do not produce some ground truth labels
2. Some small edge veins are hardly visible to the human eye. In our experience, the algorithm can segment these vessels but with a lower probability. The aim is to use this probability of every pixel as a vessel to produce better outcomes, which can be used in computer-aided diagnosis.

For this purpose, the probability slicer was implemented to see the change in the output concerning the given probability. Note, if you run the pipeline, the following image will be displayed, but `plotPredictedImgSlicer` must be run again from *cmd*. It uses `matplotlib.widgets`, which are not in our control. The same holds for the following image `plotHistogramImgSlicer`.



The results of `plotHistogramImgSlicer`, the user can see the histogram of probabilities for a given image and set the range, which is displayed.

Colorized results in original image after 2 epochs; see next image. In two epochs, with blending and cleaning small unconnected segments, the segmentation algorithm can produce a decent result.

The following image shows the final result after some postprocessing:
- Blending - for blending, the external library  was used, it was slightly modified, and some minor bugs were removed, see  [Smoothly-Blend-Image-Patches](https://github.com/Vooban/Smoothly-Blend-Image-Patches)
- Probability value - only the probability value higher than 0.8 is displayed, i.e., it displays only the segmented area where the is vessel by higher probability (against label `np.argmax`) by the algorithm
- Threshold - the segmented areas of size smaller than `img.height * img.width * threshold` is removed from the image. It removes small spots. *Segmentation algorithm has no clue that some vessels must be connected. It just sees a change in the retina image and chooses whether it is a vessel (maybe GANs can help. We will see:))*.






                  


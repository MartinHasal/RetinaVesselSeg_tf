# RetinaVesselSeg_tf: Retinal/Fundus vessel segmentation by TensorFlow

Segmenatation is based on three datasets:

- [DRIVE](http://www.isi.uu.nl/Research/Databases/DRIVE/),
- [STARE](http://www.ces.clemson.edu/ahoover/stare/),
- [CHASE_DB1](https://blogs.kingston.ac.uk/retinal/chasedb1/) 
 
Note, **datasets are in `.gitignore`** use the following link to get data: [datasets](https://www.dropbox.com/sh/kdzm2nyz3kts9hj/AACa_IFprmYlwzS-UucQ25OSa?dl=0).
After that, add datasets to root folder `RetinaVesselSeg_tf`

## Workflow
1. clone main branch
2. Download the data to main
3. run command  `python .\data_processing.py -s` in main folder. It creates `data_paths.csv` file with structure `NAME, DATASET_NAME, PATH_TO_ORIGINAL_IMAGE, MASK`  


ToDo: images, masks to tfrecods                     


This repository contains code that was a part of the project "SEGMENTATION AND CLASSIFICATION OF SKIN LESION IMAGES USING DEEP LEARNING BASED TECHNIQUES", published as part of my final year project. The current state contains the most refined tune of Segmentation networks trained on the ISIC 2016 dataset. The pipeline uses a U-Net architecture with a DenseNet201 backbone. All the hyperparameters are chosen after extensive experimentation.

To start off, you can download the "segmentation-unet-densenet121-2016-original.ipynb" file alongwith the "fetch_data.py" and "visualize.py" dependencies into the same directory. If you do not have the ISIC 2016 datasets, you can download it from https://challenge.isic-archive.com/data/. 

To resize the images into the required size you can use "resize_for_segmentation.py" (This will save you the overhead of resizing images inline during training). 

To create augmented versions of the image (to enlarge the dataset) you can use the "Image_augmentation.py". Or you can download the augmented dataset from  https://kaggle.com/datasets/99f32b1ab4c641f57789323a1e92c669ab213f7d1ca9ca8f05be854ab07f09c1. 

To use a pre-trained model, you can download the h5 file from https://www.kaggle.com/datasets/devbhuyan/extend-model and use "segmentation-unet-densenet121-2016-original (1).ipynb" to load the weights and test it on your own dataset.

Feel free to play around with the code and make sure to change the directory variables inside the codes.

    
  -------------   ALL the functions have been implemented in matlab and python both   --------------------------
   1. Augment_img_and_mask_simultaneously (Augmentation)

        --> This code imports the necessary libraries for image data augmentation using Keras and scikit-image libraries. It then sets the directory paths for the input images and corresponding masks.

        --> The code then reads in the lists of image and mask filenames in their respective directories and determines the length of the image list. The variable "rem" is set to the difference between the expected number of images (1372) and the actual number of images in the directory.

        --> The code then loops through the image and mask lists, and for each image file, it reads in the image and its corresponding mask. It then sets the values of x, y, z, and w based on the remainder of the current index divided by 4. These values will be used to augment the images by shifting, shearing, and zooming.

        --> The code then creates an ImageDataGenerator object with the specified augmentation parameters and loops through the generated images to save them to the input image directory with a new filename prefix 'AUG_'. It also generates augmented masks with the same augmentation parameters and saves them to the mask directory with the same filename prefix.
         Finally, the code decrements the value of rem and breaks out of the loop if it reaches 0 or if all images have been augmented.


  2. Semantic (semantic segmentation using U-NET architecture)
        --> This code sets up a segmentation task using U-Net architecture for the ISIC 2017 dataset.

        --> The code sets the directory paths for the training and testing sets, and then reads in the image and mask data using the imageDatastore and pixelLabelDatastore functions from MATLAB's Computer Vision Toolbox. Next, the training and testing data are combined into a single pixelLabelImageDatastore object, which combines the image and corresponding pixel-level mask data.

        -->  The code then sets up a parallel pool to speed up the training process, defines the input image size and number of output classes, and creates the U-Net layers using the unetLayers function from the Computer Vision Toolbox.

        -->  The code then defines the training options using the trainingOptions function, including the optimizer, initial learning rate, maximum number of epochs, batch size, and verbose output.

        -->  The trainNetwork function is then used to train the U-Net on the training data with the specified training options, and the resulting trained network is stored in the trainedUNet variable.

        Finally, the code uses the predict function to generate predictions for the testing data using the trained network, and the parallel pool is deleted.
 
 3. evaluateMetrics.m
       --> This code defines a MATLAB function named "evaluateMetrics" that takes two arguments "YPred" and "YTrue", which represent the predicted and true binary labels, respectively.

       -->  First, the function converts the predicted and true binary masks to logical arrays using a threshold of 0.5. Then, the function calculates the number of true positive (TP), false positive (FP), and false negative (FN) pixels.

       -->  Using these values, the function calculates the Intersection over Union (IoU) score, which is the ratio of TP to the total number of pixels that were predicted as positive. The IoU score is used to evaluate the performance of image segmentation models.

       Finally, the function calculates the F1 score, which is a weighted average of precision and recall. The F1 score is commonly used to evaluate the performance of classification models.
       

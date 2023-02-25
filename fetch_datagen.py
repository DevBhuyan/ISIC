#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

input_size = 0

def fetch(base_dir, inp_size, grp, batch_size):
    global input_size 
    input_size = inp_size
    train_dir = base_dir+'Train/'
    test_dir = base_dir+'Test/'
    
    if grp:
        train_images = train_dir+'IMAGES/'+grp+'/'
        train_masks = train_dir+'MASKS/'+grp+'/'
        
        test_images = test_dir+'IMAGES/'+grp+'/'
        test_masks = test_dir+'MASKS/'+grp+'/'
    else:
        train_images = train_dir+'IMAGES/'
        train_masks = train_dir+'MASKS/'
        
        test_images = test_dir+'IMAGES/'
        test_masks = test_dir+'MASKS/'
        
    print('train_images: ', train_images)
    print('no. of training images: ', len(os.listdir(train_images)))
    print('train_masks: ', train_masks)
    print('no. of training masks: ', len(os.listdir(train_masks)))
    print('test_images: ', test_images)
    print('no. of test images: ', len(os.listdir(test_images)))
    print('test_masks: ', test_masks)
    print('no. of test masks: ', len(os.listdir(test_masks)))
        
    train_dataset = create_generator(train_images, train_masks, batch_size, shuffle = True)
    
    val_dataset = create_generator(test_images, test_masks, 1, shuffle = True)
    
    return train_dataset, val_dataset

def create_generator(images, masks, batch_size, shuffle = True):
    global input_size
    
    datagen = ImageDataGenerator(rescale=1./255)
    
    image_generator = datagen.flow_from_directory(
        images,
        target_size=input_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=True,
        seed=42)

    mask_generator = datagen.flow_from_directory(
        masks,
        target_size=input_size,
        batch_size=batch_size,
        class_mode=None,
        color_mode='grayscale',
        shuffle=True,
        seed=42)
    
    dataset = zip(image_generator, mask_generator)
    
    return dataset    
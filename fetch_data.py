#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf

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
        
    train_img_paths = []
    train_mask_paths = []
    for file in os.listdir(train_images):
        train_img_paths.append(train_images+file)
    for file in os.listdir(train_masks):
        train_mask_paths.append(train_masks+file)
    train_dataset = create_dataset(train_img_paths, train_mask_paths, batch_size, shuffle = False)
    
    val_img_paths = []
    val_mask_paths = []
    for file in os.listdir(test_images):
        val_img_paths.append(test_images+file)
    for file in os.listdir(test_masks):
        val_mask_paths.append(test_masks+file)
    val_dataset = create_dataset(val_img_paths, val_mask_paths, 1, shuffle = False)
    
    return train_dataset, val_dataset

def load_set(img_path, mask_path):
    global input_size
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(img))
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, input_size)
    # img = img/255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.image.resize(mask, input_size)

    return img, mask

def create_dataset(img_paths, mask_paths, batch_size, shuffle = True):
    global input_size
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths)).map(load_set)
    
    if shuffle:
        dataset = dataset.shuffle(len(img_paths))
        
    dataset = dataset.batch(batch_size)
    
    return dataset    
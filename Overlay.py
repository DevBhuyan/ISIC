#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os

base_dir = '/mnt/C654078D54078003/Curious Dev B/PROJECT STAGE - II/ISIC 2016 for segmentation/'

train_dir = base_dir+'Train/'
test_dir = base_dir+'Test/'

train_images = train_dir+'IMAGES/'
train_masks = train_dir+'MASKS/'
train_seg = train_dir+'SEGMENTED/'

test_images = test_dir+'IMAGES/'
test_masks = test_dir+'MASKS/'
test_seg = test_dir+'SEGMENTED/'

for file in os.listdir(test_images):
    print(file)
    img = cv2.imread(test_images+file)
    mask = cv2.imread(test_masks+file, cv2.IMREAD_GRAYSCALE)
    (thresh, mask2) = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    res = cv2.bitwise_and(img, img, mask = mask2)
    cv2.imwrite(test_seg+file, res)



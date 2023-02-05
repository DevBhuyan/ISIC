from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import os
import numpy as np

img_dir = '/media/dev/WinD/Curious Dev B/PROJECT STAGE - II/ISIC2017/IMAGES/Train/M/'
mask_dir = '/media/dev/WinD/Curious Dev B/PROJECT STAGE - II/ISIC2017/MASKS/Train/M/'
print(img_dir)
print(mask_dir)

# M has 374 and SK has 254

img_list = os.listdir(img_dir)
mask_list = os.listdir(mask_dir)
l = len(img_list)
rem = 1372 - l

for i in range(l):
    if rem and img_list[i][:3] != 'AUG':
        img_file = img_list[i]
        mask_file = mask_list[i]
        img = io.imread(img_dir+img_file)
        img = np.expand_dims(img, axis=0)
        mask = io.imread(mask_dir+mask_file)
        mask = mask.reshape((1, ) + mask.shape + (1, ))
        c = 0
        if i%4 == 0:
            x = 0.2
            y = z = w = 0
        elif i%4 == 1:
            y = 0.2
            x = z = w = 0
        elif i%4 == 2:
            z = 0.2
            x = y = w = 0
        else:
            w = 0
            x = y = z = 0
        datagen = ImageDataGenerator(
            width_shift_range=x,  
            height_shift_range=y,    
            shear_range=z,        
            zoom_range=w)
        for batch in datagen.flow(img,
                                  batch_size=1,
                                  save_to_dir=img_dir,
                                  save_prefix='AUG_'+img_file+'_augmented_',
                                  save_format='jpg'):
            break
        datagen = ImageDataGenerator(
            width_shift_range=x,  
            height_shift_range=y,    
            shear_range=z,        
            zoom_range=w,
            fill_mode='constant', cval=0)
        for batch in datagen.flow(mask,
                                  batch_size=1,
                                  save_to_dir=mask_dir,
                                  save_prefix='AUG_'+img_file+'_augmented_',
                                  save_format='jpg'):
            break
    else:
        break
    rem -= 1
    
    
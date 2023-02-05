import os
import shutil

isic_dir = '/media/dev/WinD/Curious Dev B/PROJECT STAGE - II/ISIC2017/'
img_dir = isic_dir+'IMAGES/'
mask_dir = isic_dir+'MASKS/'

img_train_dir = img_dir+'Train/'
img_train_b = img_train_dir+'B/'
img_train_m = img_train_dir+'M/'
img_train_sk = img_train_dir+'SK/'

img_test_dir = img_dir+'Test/'
img_test_b = img_test_dir+'B/'
img_test_m = img_test_dir+'M/'
img_test_sk = img_test_dir+'SK/'

mask_train_dir = mask_dir+'Train/'
mask_test_dir = mask_dir+'Test/'

file_list = list(filter(lambda x: '.jpg' in x, os.listdir(img_test_b)))
ct = 0
for file in list(filter(lambda x: '.jpg' in x, os.listdir(mask_test_dir))):
    if file in file_list:
        print(file)
        #move mask_train_dir/file to mask_train_dir/B/file
        src = mask_test_dir+file
        target = mask_test_dir+'B/'+file
        shutil.copyfile(src, target)
        ct+=1
print(ct)
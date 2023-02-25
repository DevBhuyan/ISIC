import os
import shutil

isic_dir = '/mnt/C654078D54078003/Curious Dev B/PROJECT STAGE - II/ISIC 2016 original/'
train_dir = isic_dir+'Train/'
test_dir = isic_dir+'Test/'

img_train_dir = train_dir+'IMAGES/'
img_train_ben = img_train_dir+'ben/'
img_train_mel = img_train_dir+'mel/'
# img_train_sk = img_train_dir+'SK/'

img_test_dir = test_dir+'IMAGES/'
img_test_ben = img_test_dir+'ben/'
img_test_mel = img_test_dir+'mel/'
# img_test_sk = img_test_dir+'SK/'

seg_dir = '/mnt/C654078D54078003/Curious Dev B/PROJECT STAGE - II/ISIC 2016 segmented/'
seg_train_dir = seg_dir+'Train/'
seg_test_dir = seg_dir+'Test/'

file_list = list(filter(lambda x: '.jpg' in x, os.listdir(img_test_ben)))
ct = 0
for file in list(filter(lambda x: '.jpg' in x, os.listdir(seg_test_dir))):
    if file in file_list:
        print(file)
        src = seg_test_dir+file
        target = seg_test_dir+'ben/'+file
        shutil.copyfile(src, target)
        ct+=1
print(ct)
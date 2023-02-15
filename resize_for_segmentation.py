import os
from PIL import Image

base_dir = '/media/dev/WinD/Curious Dev B/PROJECT STAGE - II/ISIC 2017 for segmentation (python)/'

for folder in os.listdir(base_dir):
    sub_dir = base_dir+folder+'/'
    for subfolder in os.listdir(sub_dir):
        subsub_dir = sub_dir+subfolder+'/'
        for file in os.listdir(subsub_dir):
            os.chdir(subsub_dir)
            img = Image.open(file)
            if img.size != (224, 224):
                img = img.resize((224, 224))
                img.save(file)

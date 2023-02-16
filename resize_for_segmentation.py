import os
from PIL import Image

base_dir = '/media/dev/WinD/Curious Dev B/PROJECT STAGE - II/ISIC 2017 classwise segmentation/'

for folder in os.listdir(base_dir):
    sub_dir = base_dir+folder+'/'
    for subfolder in os.listdir(sub_dir):
        subsub_dir = sub_dir+subfolder+'/'
        for subsubfolder in os.listdir(subsub_dir):
            subsubsub_dir = subsub_dir+subsubfolder+'/'
        for file in os.listdir(subsubsub_dir):
            os.chdir(subsubsub_dir)
            img = Image.open(file)
            if img.size != (224, 224):
                img = img.resize((224, 224))
                img.save(file)

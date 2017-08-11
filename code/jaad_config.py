import os

# Where JAAD data will be saved if you run process_jaad.py
DATA_DIR = '/local_home/JAAD_Dataset/'

RESIZED_IMGS_DIR = '/local_home/JAAD_Dataset/resized_imgs_256'
if not os.path.exists(RESIZED_IMGS_DIR):
    os.mkdir(RESIZED_IMGS_DIR)
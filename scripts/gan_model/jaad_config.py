import os

# Where JAAD data will be saved if you run process_jaad.py
DATA_DIR = '/home/pratik/DeepIntent_Datasets/JAAD_Dataset/'

RESIZED_IMGS_DIR = '/home/pratik/DeepIntent_Datasets/JAAD_Dataset/resized_imgs_128'
if not os.path.exists(RESIZED_IMGS_DIR):
    os.mkdir(RESIZED_IMGS_DIR)
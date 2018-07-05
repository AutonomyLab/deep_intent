import os

# Where JAAD data will be saved if you run process_jaad.py
DATA_DIR = '/local_home/JAAD_Dataset/'

XML_DIR = '/local_home/JAAD_Dataset/JAAD_behavioral_data_xml_new/'

RESIZED_IMGS_DIR = '/local_home/JAAD_Dataset/iros/resized_imgs_208_thesis/'
if not os.path.exists(RESIZED_IMGS_DIR):
    os.mkdir(RESIZED_IMGS_DIR)

TEST_DIR = '/local_home/JAAD_Dataset/iros/resized_imgs_208_thesis/test/'
if not os.path.exists(TEST_DIR):
    os.mkdir(TEST_DIR)

TRAIN_DIR = '/local_home/JAAD_Dataset/iros/resized_imgs_208_thesis/train/'
if not os.path.exists(TRAIN_DIR):
    os.mkdir(TRAIN_DIR)

VAL_DIR = '/local_home/JAAD_Dataset/iros/resized_imgs_208_thesis/val/'
if not os.path.exists(VAL_DIR):
    os.mkdir(VAL_DIR)

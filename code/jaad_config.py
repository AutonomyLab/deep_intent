import os

# Where JAAD data will be saved if you run process_jaad.py
DATA_DIR = '/local_home/JAAD_Dataset/'

XML_DIR = '/local_home/JAAD_Dataset/JAAD_behavioral_data_xml/'

RESIZED_IMGS_DIR = '/local_home/JAAD_Dataset/resized_imgs_128'
if not os.path.exists(RESIZED_IMGS_DIR):
    os.mkdir(RESIZED_IMGS_DIR)

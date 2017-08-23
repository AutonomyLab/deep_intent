import os

# Where KITTI data will be saved if you run process_kitti.py
# If you directly download the processed data, change to the path of the data.
DATA_DIR = '/local_home/KITTI_Dataset/'

# Where model weights and config will be saved if you run kitti_train.py
# If you directly download the trained weights, change to appropriate path.
WEIGHTS_DIR = './model_data/'

# Where results (prediction plots and evaluation file) will be saved.
RESULTS_SAVE_DIR = './kitti_results/'

RESIZED_IMGS_DIR = '/local_home/KITTI_Dataset/resized_imgs_128'
if not os.path.exists(RESIZED_IMGS_DIR):
    os.mkdir(RESIZED_IMGS_DIR)
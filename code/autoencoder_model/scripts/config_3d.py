from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.optimizers import SGD
from keras.optimizers import Adam
import os

# -------------------------------------------------
# Background config:
# DATA_DIR= '/home/pratik/DeepIntent_Datasets/KITTI_Dataset/'
DATA_DIR= '/local_home/JAAD_Dataset/resized_imgs_128/train/'
# DATA_DIR= '/local_home/data/KITTI_data/'

MODEL_DIR = './../models'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

CHECKPOINT_DIR = './../checkpoints'
if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

GEN_IMAGES_DIR = './../generated_images'
if not os.path.exists(GEN_IMAGES_DIR):
    os.mkdir(GEN_IMAGES_DIR)

LOG_DIR = './../logs'
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

TF_LOG_DIR = "./../tf_logs"
if not os.path.exists(TF_LOG_DIR):
    os.mkdir(TF_LOG_DIR)

PRINT_MODEL_SUMMARY = True
SAVE_MODEL = False
SAVE_GENERATED_IMAGES = True
SHUFFLE = True
VIDEO_LENGTH = 10
IMG_SIZE = (128, 128, 3)

# -------------------------------------------------
# Network configuration:
print ("Loading network/training configuration...")

BATCH_SIZE = 10
NB_EPOCHS = 100

OPTIM = Adam(lr=0.0001, beta_1=0.5)
# OPTIM = SGD(lr=0.00001, momentum=0.5, nesterov=True)


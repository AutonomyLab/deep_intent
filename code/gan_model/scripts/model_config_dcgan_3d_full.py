from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.optimizers import SGD
from keras.optimizers import Adam
import os

# -------------------------------------------------
# Background config:
# DATA_DIR= '/home/pratik/DeepIntent_Datasets/KITTI_Dataset/'
# DATA_DIR= '/local_home/data/JAAD_data/'
# DATA_DIR= '/local_home/data/KITTI_data/'
# DATA_DIR = '/grad/2/pgujjar/DeepIntent/data/JAAD_data'
DATA_DIR = './../data/KITTI_data'
# DATA_DIR = './../data/JAAD_data'

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
DATA_AUGMENTATION = False
SHUFFLE = True
VIDEO_LENGTH = 32

# -------------------------------------------------
# Network configuration:
print ("Loading network/training configuration...")

BATCH_SIZE = 4
NB_EPOCHS = 1000
IMAGE_SHAPE = (64, 64, 3)

# g_optim = SGD(lr=0.0001, momentum=0.5, nesterov=True)
# d_optim = Adam(lr=0.005, beta_1=0.5)
G_OPTIM = Adam(lr=0.001, beta_1=0.5)
D_OPTIM = SGD(lr=0.001, momentum=0.5, nesterov=True)

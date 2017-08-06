from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.optimizers import SGD
from keras.optimizers import Adam
import os

# -------------------------------------------------
# Background config:
# DATA_DIR= '/home/pratik/DeepIntent_Datasets/KITTI_Dataset/'
DATA_DIR= '/local_home/data/JAAD_data/'

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

PRINT_MODEL_SUMMARY = False
SAVE_MODEL = False
SAVE_GENERATED_IMAGES = True
SHUFFLE = True

# -------------------------------------------------
# Network configuration:
print ("Loading network/training configuration...")

BATCH_SIZE = 128
NB_EPOCHS = 100

OPTIM = Adam(lr=0.00001, beta_1=0.5)


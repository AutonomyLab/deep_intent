from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.optimizers import SGD
import os

# -------------------------------------------------
# Background config:
DATA_DIR= '/home/pratik/DeepIntent_Datasets/KITTI_Dataset/'

MODEL_DIR = './models'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

CHECKPOINT_DIR = './checkpoints'
if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

PRINT_MODEL_SUMMARY = False
SAVE_MODEL = False
SAVE_GENERATED_IMAGES = True
DATA_AUGMENTATION = False

# -------------------------------------------------
# Network configuration:
print ("Loading network/training configuration...")

BATCH_SIZE = 128
NB_EPOCHS = 10

lr_schedule = [60, 120, 160]  # epoch_step

def schedule(epoch_idx):
    if (epoch_idx + 1) < lr_schedule[0]:
        return 0.1
    elif (epoch_idx + 1) < lr_schedule[1]:
        return 0.02  # lr_decay_ratio = 0.2
    elif (epoch_idx + 1) < lr_schedule[2]:
        return 0.004
    return 0.0008


sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
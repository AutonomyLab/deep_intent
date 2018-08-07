from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras import backend as K
K.set_image_dim_ordering('tf')
import socket
import os

# -------------------------------------------------
# Background config:
hostname = socket.gethostname()
if hostname == 'baymax':
    path_var = 'baymax/'
elif hostname == 'walle':
    path_var = 'walle/'
elif hostname == 'bender':
    path_var = 'bender/'
else:
    path_var = 'zhora/'

DATA_DIR= '/local_home/JAAD_Dataset/iros/resized_imgs_208_thesis/train/'
# DATA_DIR= '/local_home/data/KITTI_data/'

TEST_DATA_DIR= '/local_home/JAAD_Dataset/iros/resized_imgs_208_thesis/test/'

VAL_DATA_DIR= '/local_home/JAAD_Dataset/iros/resized_imgs_208_thesis/val/'

PRETRAINED_C3D= '/home/pratik/git_projects/c3d-keras/models/sports1M_weights_tf.json'
PRETRAINED_C3D_WEIGHTS= '/home/pratik/git_projects/c3d-keras/models/sports1M_weights_tf.h5'

RESULTS_DIR = '/local_home/JAAD_Dataset/thesis/results/baselineCla/'

MODEL_DIR = RESULTS_DIR + 'models'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

CHECKPOINT_DIR = RESULTS_DIR + 'checkpoints'
if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

GEN_IMAGES_DIR = RESULTS_DIR + 'generated_images'
if not os.path.exists(GEN_IMAGES_DIR):
    os.mkdir(GEN_IMAGES_DIR)

CLA_GEN_IMAGES_DIR = GEN_IMAGES_DIR + '/cla_gen/'
if not os.path.exists(CLA_GEN_IMAGES_DIR):
    os.mkdir(CLA_GEN_IMAGES_DIR)

LOG_DIR = RESULTS_DIR + 'logs'
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

TF_LOG_DIR = RESULTS_DIR + 'tf_logs'
if not os.path.exists(TF_LOG_DIR):
    os.mkdir(TF_LOG_DIR)

TF_LOG_CLA_DIR = RESULTS_DIR + 'tf_cla_logs'
if not os.path.exists(TF_LOG_CLA_DIR):
    os.mkdir(TF_LOG_CLA_DIR)

TEST_RESULTS_DIR = RESULTS_DIR + 'test_results/'
if not os.path.exists(TEST_RESULTS_DIR):
    os.mkdir(TEST_RESULTS_DIR)

PRINT_MODEL_SUMMARY = True
SAVE_MODEL = True
PLOT_MODEL = True
SAVE_GENERATED_IMAGES = True
SHUFFLE = True
VIDEO_LENGTH = 16
IMG_SIZE = (128, 208, 3)
CLASSIFIER = True
LOSS_WEIGHTS = [1, 1]
RAM_DECIMATE = True
RETRAIN_CLASSIFIER = True

CLASS_TARGET_INDEX = 8

ROT_MAX = 5
SFT_H_MAX = 0.02
SFT_V_MAX = 0.02
ZOOM_MAX = 0.2
BRIGHT_RANGE_L = 0.5
BRIGHT_RANGE_H = 1.5

ped_actions = ['slow down', 'standing', 'walking', 'speed up', 'nod', 'unknown',
               'clear path', 'handwave', 'crossing', 'looking', 'no ped']

simple_ped_set = ['crossing']

# -------------------------------------------------
# Network configuration:
print ("Loading network/training configuration...")
print ("Config file: " + str(__name__))

BATCH_SIZE = 15
TEST_BATCH_SIZE = 1
NB_EPOCHS_CLASS = 30

# OPTIM_C = Adam(lr=0.0000002, beta_1=0.5)
# OPTIM_C = SGD(lr=0.0001, momentum=0.9, nesterov=True)
OPTIM_C = RMSprop(lr=0.0001, rho=0.9)

# lr_schedule = [10, 20, 30]  # epoch_step

# def schedule(epoch_idx):
#     if (epoch_idx + 1) < lr_schedule[0]:
#         return 0.00000001
#     elif (epoch_idx + 1) < lr_schedule[1]:
#         return 0.000000001  # lr_decay_ratio = 10
#     elif (epoch_idx + 1) < lr_schedule[2]:
#         return 0.000000001
#     return 0.000000001


lr_schedule = [7, 16, 30, 30]  # epoch_step

def schedule(epoch_idx):
    if (epoch_idx) <= lr_schedule[0]:
        return 0.00001
    elif (epoch_idx) <= lr_schedule[1]:
        return 0.000001  # lr_decay_ratio = 10
    elif (epoch_idx) <= lr_schedule[2]:
        return 0.0000001  # lr_decay_ratio = 10
    elif (epoch_idx) <= lr_schedule[3]:
        return 0.0000001
    return 0.0000001
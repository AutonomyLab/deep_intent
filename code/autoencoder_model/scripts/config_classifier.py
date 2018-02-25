from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.optimizers import SGD
from keras.optimizers import Adam
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

DATA_DIR= '/local_home/JAAD_Dataset/iros/resized_imgs_128/train/'
# DATA_DIR= '/local_home/data/KITTI_data/'

TEST_DATA_DIR= '/local_home/JAAD_Dataset/iros/resized_imgs_128/test/'

VAL_DATA_DIR= '/local_home/JAAD_Dataset/iros/resized_imgs_128/val/'

PRETRAINED_C3D= '/home/pratik/git_projects/c3d-keras/models/sports1M_weights_tf.json'
PRETRAINED_C3D_WEIGHTS= '/home/pratik/git_projects/c3d-keras/models/sports1M_weights_tf.h5'

MODEL_DIR = './../' + path_var + 'models'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

CHECKPOINT_DIR = './../' + path_var + 'checkpoints'
if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

GEN_IMAGES_DIR = './../' + path_var + 'generated_images'
if not os.path.exists(GEN_IMAGES_DIR):
    os.mkdir(GEN_IMAGES_DIR)

CLA_GEN_IMAGES_DIR = GEN_IMAGES_DIR + '/cla_gen/'
if not os.path.exists(CLA_GEN_IMAGES_DIR):
    os.mkdir(CLA_GEN_IMAGES_DIR)

ATTN_WEIGHTS_DIR = './../' + path_var + 'attn_weights'
if not os.path.exists(ATTN_WEIGHTS_DIR):
    os.mkdir(ATTN_WEIGHTS_DIR)

LOG_DIR = './../' + path_var + 'logs'
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

TF_LOG_DIR = './../' + path_var + 'tf_logs'
if not os.path.exists(TF_LOG_DIR):
    os.mkdir(TF_LOG_DIR)

TF_LOG_CLA_DIR = './../' + path_var + 'tf_cla_logs'
if not os.path.exists(TF_LOG_CLA_DIR):
    os.mkdir(TF_LOG_CLA_DIR)

TEST_RESULTS_DIR = './../' + path_var + 'test_results'
if not os.path.exists(TEST_RESULTS_DIR):
    os.mkdir(TEST_RESULTS_DIR)

PRINT_MODEL_SUMMARY = True
SAVE_MODEL = True
PLOT_MODEL = True
SAVE_GENERATED_IMAGES = True
SHUFFLE = True
VIDEO_LENGTH = 32
IMG_SIZE = (128, 128, 3)
VIS_ATTN = True
ATTN_COEFF = 0
# KL coeff damages learning
KL_COEFF = 0
CLASSIFIER = True
RAM_DECIMATE = True
RETRAIN_CLASSIFIER = True
CLASS_TARGET_INDEX = 24

ped_actions = ['slow down', 'moving slow', 'standing', 'stopped',
               'speed up', 'moving fast', 'look', 'looking',  'clear path',
               'crossing', 'nod', 'handwave', 'unknown']

simple_ped_set = ['crossing', 'stopped', 'looking', 'clear path', 'unknown']

driver_actions = ['moving slow', 'slowing down', 'standing', 'speeding up', 'moving fast']
simple_driver_set = ['slow down', 'stop', 'speed up']

joint_action_set = ['moving slow', 'slowing down', 'standing', 'speeding up', 'moving fast',
                    'slow down', 'standing', 'moving fast', 'speed up', 'look', 'nod', 'unknown',
                    'moving slow', 'flasher signal', 'looking' , 'handwave', 'clear path',
                    'stopped', 'slowing down', 'crossing', 'speeding up']

formatted_joint_action_set = ['car moving slow', 'car slowing down', 'car standing', 'car speeding up', 'car moving fast',
                              'ped slow down', 'ped standing', 'ped moving fast', 'ped speed up', 'ped look',
                              'ped nod', 'ped unknown', 'ped moving slow', 'ped flasher signal', 'ped looking' ,
                              'ped handwave', 'ped clear path', 'ped stopped', 'ped slowing down', 'ped crossing',
                              'ped speeding up']


# -------------------------------------------------
# Network configuration:
print ("Loading network/training configuration...")
print ("Config file: " + str(__name__))

BATCH_SIZE = 25
NB_EPOCHS_AUTOENCODER = 0
NB_EPOCHS_CLASS = 100

OPTIM_A = Adam(lr=0.0001, beta_1=0.5)
# OPTIM_C = Adam(lr=0.0000002, beta_1=0.5)
OPTIM_C = SGD(lr=0.0001, momentum=0.9, nesterov=True)


auto_lr_schedule = [25, 30, 35]  # epoch_step
def auto_schedule(epoch_idx):
    if (epoch_idx + 1) < auto_lr_schedule[0]:
        return 0.0001
    elif (epoch_idx + 1) < auto_lr_schedule[1]:
        return 0.00001  # lr_decay_ratio = 10
    elif (epoch_idx + 1) < auto_lr_schedule[2]:
        return 0.00001
    return 0.00001


clas_lr_schedule = [10, 11, 12]  # epoch_step
def clas_schedule(epoch_idx):
    if (epoch_idx + 1) < clas_lr_schedule[0]:
        return 0.0001
    elif (epoch_idx + 1) < clas_lr_schedule[1]:
        return 0.00001  # lr_decay_ratio = 10
    elif (epoch_idx + 1) < clas_lr_schedule[2]:
        return 0.000001
    return 0.0000001
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import adadelta
from keras.optimizers import rmsprop
from keras.layers import Layer
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

HD_DATA_DIR= '/local_home/JAAD_Dataset/iros/resized_imgs_256/train/'

VAL_DATA_DIR= '/local_home/JAAD_Dataset/iros/resized_imgs_128/val/'

VAL_HD_DATA_DIR= '/local_home/JAAD_Dataset/iros/resized_imgs_256/val/'

TEST_DATA_DIR= '/local_home/JAAD_Dataset/iros/resized_imgs_128/test/'

MODEL_DIR = './../' + path_var + 'models'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

CHECKPOINT_DIR = './../' + path_var + 'checkpoints'
if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

ATTN_WEIGHTS_DIR = './../' + path_var + 'attn_weights'
if not os.path.exists(ATTN_WEIGHTS_DIR):
    os.mkdir(ATTN_WEIGHTS_DIR)

GEN_IMAGES_DIR = './../' + path_var + 'generated_images'
if not os.path.exists(GEN_IMAGES_DIR):
    os.mkdir(GEN_IMAGES_DIR)

CLA_GEN_IMAGES_DIR = GEN_IMAGES_DIR + '/cla_gen/'
if not os.path.exists(CLA_GEN_IMAGES_DIR):
    os.mkdir(CLA_GEN_IMAGES_DIR)

LOG_DIR = './../' + path_var + 'logs'
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

TF_LOG_DIR = './../' + path_var + 'tf_logs'
if not os.path.exists(TF_LOG_DIR):
    os.mkdir(TF_LOG_DIR)

TF_LOG_GAN_DIR = './../' + path_var + 'tf_gan_logs'
if not os.path.exists(TF_LOG_GAN_DIR):
    os.mkdir(TF_LOG_GAN_DIR)

TEST_RESULTS_DIR = './../' + path_var + 'test_results'
if not os.path.exists(TEST_RESULTS_DIR):
    os.mkdir(TEST_RESULTS_DIR)

PRINT_MODEL_SUMMARY = True
SAVE_MODEL = True
PLOT_MODEL = True
SAVE_GENERATED_IMAGES = True
SHUFFLE = True
VIDEO_LENGTH = 30
IMG_SIZE = (128, 128, 3)
ADVERSARIAL = False
BUF_SIZE = 10
LOSS_WEIGHTS = [1, 1]
ATTN_COEFF = 0
KL_COEFF = 0

# -------------------------------------------------
# Network configuration:
print ("Loading network/training configuration.")
print ("Config file: " + str(__name__))

BATCH_SIZE = 7
NB_EPOCHS_AUTOENCODER = 30
NB_EPOCHS_GAN = 0

OPTIM_A = Adam(lr=0.0001, beta_1=0.5)
OPTIM_G = Adam(lr=0.00001, beta_1=0.5)
# OPTIM_D = Adam(lr=0.000001, beta_1=0.5)
# OPTIM_D = SGD(lr=0.000001, momentum=0.5, nesterov=True)
OPTIM_D = rmsprop(lr=0.000001)

lr_schedule = [10, 20, 30]  # epoch_step

def schedule(epoch_idx):
    if (epoch_idx + 1) < lr_schedule[0]:
        return 0.0001
    elif (epoch_idx + 1) < lr_schedule[1]:
        return 0.0001  # lr_decay_ratio = 10
    elif (epoch_idx + 1) < lr_schedule[2]:
        return 0.00001
    return 0.000001



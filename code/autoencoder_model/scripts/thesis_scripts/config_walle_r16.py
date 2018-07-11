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

DATA_DIR= '/local_home/JAAD_Dataset/iros/resized_imgs_208_thesis/train/'

VAL_DATA_DIR= '/local_home/JAAD_Dataset/iros/resized_imgs_208_thesis/val/'

TEST_DATA_DIR= '/local_home/JAAD_Dataset/iros/resized_imgs_208_thesis/test/'
# TEST_DATA_DIR= '/local_home/JAAD_Dataset/fun_experiments/resized/'

RESULTS_DIR = '/local_home/JAAD_Dataset/thesis/results/NRNN16/'

# MODEL_DIR = './../' + path_var + 'models'
MODEL_DIR = RESULTS_DIR + 'models/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

# CHECKPOINT_DIR = './../' + path_var + 'checkpoints'
CHECKPOINT_DIR = RESULTS_DIR + 'checkpoints/'
if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

GEN_IMAGES_DIR = RESULTS_DIR + 'generated_images/'
if not os.path.exists(GEN_IMAGES_DIR):
    os.mkdir(GEN_IMAGES_DIR)

LOG_DIR = RESULTS_DIR + 'logs/'
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

TF_LOG_DIR = RESULTS_DIR + 'tf_logs/'
if not os.path.exists(TF_LOG_DIR):
    os.mkdir(TF_LOG_DIR)

TEST_RESULTS_DIR = RESULTS_DIR + 'test_results/'
if not os.path.exists(TEST_RESULTS_DIR):
    os.mkdir(TEST_RESULTS_DIR)

PRINT_MODEL_SUMMARY = True
SAVE_MODEL = True
PLOT_MODEL = True
SAVE_GENERATED_IMAGES = True
SHUFFLE = True
VIDEO_LENGTH = 32
IMG_SIZE = (128, 208, 3)
RAM_DECIMATE = False
REVERSE = True
FILTER_SIZE = 3


# -------------------------------------------------
# Network configuration:
print ("Loading network/training configuration.")
print ("Config file: " + str(__name__))

BATCH_SIZE = 9
TEST_BATCH_SIZE = 1
NB_EPOCHS_AUTOENCODER = 30

# OPTIM_A = Adam(lr=0.0001, beta_1=0.5)
OPTIM_A = rmsprop(lr=0.0001, rho=0.9)
OPTIM_B = rmsprop(lr=0.00001, rho=0.9)
# OPTIM_A = SGD(lr=0.000001, momentum=0.5, nesterov=True)

lr_schedule = [7, 14, 20, 30]  # epoch_step

def schedule(epoch_idx):
    if (epoch_idx) <= lr_schedule[0]:
        return 0.001
    elif (epoch_idx) <= lr_schedule[1]:
        return 0.0001  # lr_decay_ratio = 10
    elif (epoch_idx) <= lr_schedule[2]:
        return 0.00001  # lr_decay_ratio = 10
    elif (epoch_idx) <= lr_schedule[3]:
        return 0.00001
    return 0.00001




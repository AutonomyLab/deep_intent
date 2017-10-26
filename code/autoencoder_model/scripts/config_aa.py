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
else:
    path_var = ''

DATA_DIR= '/local_home/JAAD_Dataset/resized_imgs_128/train/'
# DATA_DIR= '/local_home/data/KITTI_data/'

TEST_DATA_DIR= '/local_home/JAAD_Dataset/resized_imgs_128/test/'

MODEL_DIR = './../' + path_var + 'models'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

CHECKPOINT_DIR = './../' + path_var + 'checkpoints'
if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

GEN_IMAGES_DIR = './../' + path_var + 'generated_images'
if not os.path.exists(GEN_IMAGES_DIR):
    os.mkdir(GEN_IMAGES_DIR)

LOG_DIR = './../' + path_var + 'logs'
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

TF_LOG_DIR = './../' + path_var + 'tf_logs'
if not os.path.exists(TF_LOG_DIR):
    os.mkdir(TF_LOG_DIR)

TEST_RESULTS_DIR = './../' + path_var + 'test_results'
if not os.path.exists(TEST_RESULTS_DIR):
    os.mkdir(TEST_RESULTS_DIR)

PRINT_MODEL_SUMMARY = True
SAVE_MODEL = True
PLOT_MODEL = True
SAVE_GENERATED_IMAGES = True
SHUFFLE = True
VIDEO_LENGTH = 20
IMG_SIZE = (128, 128, 3)
VIS_ATTN = True
ATTN_COEFF = 0
ADVERSARIAL = False

# -------------------------------------------------
# Network configuration:
print ("Loading network/training configuration...")

BATCH_SIZE = 10
NB_EPOCHS_AUTOENCODER = 100
NB_EPOCHS_AAE = 100

OPTIM_A = Adam(lr=0.00001, beta_1=0.5)
OPTIM_G = Adam(lr=0.0001, beta_1=0.5)
OPTIM_D = SGD(lr=0.0000001, momentum=0.5, nesterov=True)
# OPTIM = rmsprop(lr=0.00001)

lr_schedule = [10, 20, 30]  # epoch_step

def schedule(epoch_idx):
    if (epoch_idx + 1) < lr_schedule[0]:
        return 0.0001
    elif (epoch_idx + 1) < lr_schedule[1]:
        return 0.0001  # lr_decay_ratio = 10
    elif (epoch_idx + 1) < lr_schedule[2]:
        return 0.00001
    return 0.000001

# Custom loss layer
class CustomLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomLossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(CustomLossLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def attn_loss(self, a):
        attn_loss = K.mean(K.mean(K.square(1 - K.sum(a, axis=1)), axis=1), axis=1)
        return ATTN_COEFF * K.mean(attn_loss)

    def call(self, inputs):
        x = inputs
        loss = self.attn_loss(x)
        self.add_loss(loss, inputs=inputs)
        # We do use this output.
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

def broadcast_channels(x):
    return K.repeat_elements(x, 128, axis=-1)

def broadcast_output_shape(input_shape):
    return input_shape[0:3] + (128,)

def expectation(x):
    return K.sum(K.sum(x, axis=-2), axis=-2)

# def expectation_output_shape(input_shape):
#     return input_shape[0:3] + (128,)

# aclstm_1 = ConvLSTM2D(filters=1,
    #                       kernel_size=(3, 3),
    #                       dilation_rate=(2, 2),
    #                       strides=(1, 1),
    #                       padding='same',
    #                       return_sequences=True,
    #                       recurrent_dropout=0.5,
    #                       name='aclstm_1')(out_4)
    # x = TimeDistributed(BatchNormalization())(aclstm_1)
    # flat_1 = TimeDistributed(Flatten())(x)
    # dense_1 = TimeDistributed(Dense(units=64 * 64, activation='softmax',
    #                                 kernel_initializer=RandomNormal(mean=0.5, stddev=0.125)))(flat_1)
    # x = TimeDistributed(Dropout(0.5))(dense_1)
    # a_1 = Reshape(target_shape=(10, 64, 64, 1))(x)
    #
    # # aclstm_2 = ConvLSTM2D(filters=1,
    # #                       kernel_size=(3, 3),
    # #                       dilation_rate=(2, 2),
    # #                       strides=(1, 1),
    # #                       padding='same',
    # #                       return_sequences=True,
    # #                       recurrent_dropout=0.5,
    # #                       name='aclstm_2')(a_1)
    # # x = TimeDistributed(BatchNormalization())(aclstm_2)
    # # flat_2 = TimeDistributed(Flatten())(x)
    # # dense_2 = TimeDistributed(Dense(units=64 * 64, activation='softmax'))(flat_2)
    # # x = TimeDistributed(Dropout(0.5))(dense_2)
    # # a_2 = Reshape(target_shape=(10, 64, 64, 1))(x)
    #
    # x = CustomLossLayer()(a_1)
    # x = Flatten()(x)
    # x = RepeatVector(n=64)(x)
    # x = Permute((2, 1))(x)
    # x = Reshape(target_shape=(10, 64, 64, 64))(x)
    # mul_1 = multiply([out_4, x])
    # out_5 = UpSampling3D(size=(1, 2, 2))(mul_1)




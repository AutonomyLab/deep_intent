from keras.optimizers import SGD
import os

# -------------------------------------------------
# Background config:
DATA_DIR= '/home/pratik/DeepIntent_Datasets/'

MODEL_DIR = 'models'Code borrowed from PredNet (Lotter et al. 2017, https://arxiv.org/abs/1605.08104)
'''
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

CHECKPOINT_DIR = 'checkpoints'
if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

PRINT_MODEL_SUMMARY = True
SAVE_MODEL_PLOT = True
DATA_AUGMENTATION = False

# -------------------------------------------------
# Network configuration:
print ("Loading network/training configuration...")

batch_size = 10
nb_epochs = 20
lr_schedule = [60, 120, 160]  # epoch_step

# Input image dimensions
# Use grayscale video
img_rows, img_cols, img_chns = 64, 64, 3
original_image_size = (img_rows, img_cols, img_chns)

latent_dim = 2
intermediate_dim = 512
epochs = 50
epsilon_std = 1.0

# Number of convolutional filters to use
n_filters = 64
# Convolutional kernel size
n_conv = 3

def schedule(epoch_idx):
    if (epoch_idx + 1) < lr_schedule[0]:
        return 0.1
    elif (epoch_idx + 1) < lr_schedule[1]:
        return 0.02  # lr_decay_ratio = 0.2
    elif (epoch_idx + 1) < lr_schedule[2]:
        return 0.004
    return 0.0008


sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
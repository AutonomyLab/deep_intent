from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib.pyplot import axes

import hickle as hkl
import numpy as np

np.random.seed(2 ** 10)
import tensorflow as tf
import itertools
from keras import backend as K
K.set_image_dim_ordering('tf')
import matplotlib.pyplot as plt
from keras import regularizers
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers.core import Activation
from keras.utils.vis_utils import plot_model
from keras.initializers import RandomNormal
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import Conv3DTranspose
from keras.layers.convolutional import UpSampling3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.convolutional import MaxPooling3D
from keras.layers.convolutional import ZeroPadding3D
from keras.layers.merge import multiply
from keras.layers.merge import add
from keras.layers.merge import concatenate
from keras.layers.core import Permute
from keras.layers.core import RepeatVector
from keras.layers.core import Dense
from keras.layers.core import Lambda
from keras.layers.core import Reshape
from keras.layers.core import Flatten
from keras.utils import to_categorical
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import kullback_leibler_divergence
from keras.losses import mean_squared_error
from keras.layers import Input
from keras.models import Model
from keras.models import model_from_json
from keras.metrics import top_k_categorical_accuracy
from experience_memory import ExperienceMemory
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from image_utils import random_rotation
from image_utils import random_shift
from image_utils import random_zoom
from image_utils import flip_axis
from image_utils import random_brightness
from config_sigc import *
from sys import stdout

import tb_callback
import lrs_callback
import argparse
import random
import math
import cv2
import os


def process_prec3d():
    json_file = open(PRETRAINED_C3D, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    model.load_weights(PRETRAINED_C3D_WEIGHTS)
    print("Loaded weights from disk")
    for layer in model.layers[:13]:
        layer.trainable = RETRAIN_CLASSIFIER

    # for layer in model.layers[:17]:
    #     layer.trainable = RETRAIN_CLASSIFIER

    # i = 0
    # for layer in model.layers:
    #     print(layer, i)
    #     i = i + 1
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    return model


def pretrained_c3d():
    c3d = process_prec3d()
    print (c3d.summary())

    inputs = Input(shape=(16, 128, 208, 3))

    # lstm_1 = ConvLSTM2D(filters=256,
    #                     kernel_size=(3, 3),
    #                     strides=(1, 1),
    #                     padding='same',
    #                     return_sequences=True,
    #                     recurrent_dropout=0.5)(conv_10)
    # lstm_1 = TimeDistributed(BatchNormalization())(lstm_1)
    # lstm_1 = TimeDistributed(LeakyReLU(alpha=0.2))(lstm_1)
    #
    # lstm_2 = ConvLSTM2D(filters=256,
    #                     kernel_size=(3, 3),
    #                     strides=(1, 1),
    #                     padding='same',
    #                     return_sequences=True,
    #                     recurrent_dropout=0.5)(lstm_1)
    # lstm_2 = TimeDistributed(BatchNormalization())(lstm_2)
    # lstm_2 = TimeDistributed(LeakyReLU(alpha=0.2))(lstm_2)
    #
    # lstm_3 = ConvLSTM2D(filters=256,
    #                     kernel_size=(3, 3),
    #                     strides=(1, 1),
    #                     padding='same',
    #                     return_sequences=False,
    #                     recurrent_dropout=0.5)(lstm_2)
    # lstm_3 = BatchNormalization()(lstm_3)
    # lstm_3 = LeakyReLU(alpha=0.2)(lstm_3)
    #
    # resized = TimeDistributed(Lambda(lambda image: tf.image.resize_images(image, (112, 112))))(inputs)

    c3d_out = c3d(inputs)

    dense = Dense(units=1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(c3d_out)
    x = BatchNormalization()(dense)
    x = Dropout(0.5)(x)
    x = Dense(units=1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    actions = Dense(units=len(simple_ped_set), activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(x)
    # actions = Dense(units=len(simple_ped_set), activation='softmax', kernel_regularizer=regularizers.l2(0.001))(c3d_out)
    model = Model(inputs=inputs, outputs=actions)

    # i=0
    # for layer in model.layers:
    #     print (layer, i)
    #     i = i+1
    # exit(0)

    return model


def c3d_scratch():
    model = Sequential()
    model.add(Conv3D(filters=64,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='conv1',
                     input_shape=(16, 128, 208, 3)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(filters=64,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='conv2'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(filters=128,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='conv3a'))
    model.add(Conv3D(filters=256,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3'))
    # 4th layer group
    model.add(Conv3D(filters=512,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='conv4a'))
    model.add(Conv3D(filters=512,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool4'))
    # 5th layer group
    model.add(Conv3D(filters=512,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='conv5a'))
    model.add(Conv3D(filters=512,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='conv5b'))
    model.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool5'))
    model.add(Flatten())

    # FC layers group
    model.add(Dense(512, activation='relu', name='fc6'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu', name='fc7'))
    model.add(BatchNormalization())
    model.add(Dropout(.5))
    model.add(Dense(len(simple_ped_set), activation='sigmoid', name='fc8'))

    return model


def set_trainability(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def arrange_images(video_stack):
    n_frames = video_stack.shape[0] * video_stack.shape[1]
    frames = np.zeros((n_frames,) + video_stack.shape[2:], dtype=video_stack.dtype)

    frame_index = 0
    for i in range(video_stack.shape[0]):
        for j in range(video_stack.shape[1]):
            frames[frame_index] = video_stack[i, j]
            frame_index += 1

    img_height = video_stack.shape[2]
    img_width = video_stack.shape[3]
    # width = img_size x video_length
    width = img_width * VIDEO_LENGTH
    # height = img_size x batch_size
    height = img_height * BATCH_SIZE
    shape = frames.shape[1:]
    image = np.zeros((height, width, shape[2]), dtype=video_stack.dtype)
    frame_number = 0
    for i in range(BATCH_SIZE):
        for j in range(VIDEO_LENGTH):
            image[(i * img_height):((i + 1) * img_height), (j * img_width):((j + 1) * img_width)] = frames[frame_number]
            frame_number = frame_number + 1

    return image

def metric_tp(y_true, y_pred):
    value, update_op = tf.metrics.true_positives(K.cast(y_true, 'int32'), K.cast(K.round(y_pred), 'int32'))

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'tp' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def metric_precision(y_true, y_pred):
    value, update_op = tf.metrics.precision(K.cast(y_true, 'int32'), K.cast(K.round(y_pred), 'int32'))

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'precision' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def metric_recall(y_true, y_pred):
    value, update_op = tf.metrics.recall(K.cast(y_true, 'int32'), K.cast(K.round(y_pred), 'int32'))

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'recall' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def metric_mpca(y_true, y_pred):
    value, update_op = tf.metrics.mean_per_class_accuracy(K.cast(y_true, 'int32'),
                                                          K.cast(K.round(y_pred), 'int32'),
                                                          len(simple_ped_set))

    # print (tf.local_variables())
    # exit(0)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'mpca' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def load_weights(weights_file, model):
    model.load_weights(weights_file)


def run_utilities(classifier, CLA_WEIGHTS):
    if PRINT_MODEL_SUMMARY:
        if CLASSIFIER:
            print("Classifier:")
            print (classifier.summary())
        # exit(0)

    # Save model to file
    if SAVE_MODEL:
        if CLASSIFIER:
            model_json = classifier.to_json()
            with open(os.path.join(MODEL_DIR, "classifier.json"), "w") as json_file:
                json_file.write(model_json)

        if PLOT_MODEL:
            if CLASSIFIER:
                plot_model(classifier, to_file=os.path.join(MODEL_DIR, 'classifier.png'), show_shapes=True)

    if CLASSIFIER:
        if CLA_WEIGHTS != "None":
            print("Pre-loading classifier with weights.")
            load_weights(CLA_WEIGHTS, classifier)


def random_augmentation(video):
    # Toss a die
    k = np.random.randint(0, 5, dtype=int)
    if k == 0:
        for i in range(VIDEO_LENGTH):
            video[i] = (video[i].astype(np.float32) - 127.5) / 127.5
        return video

    elif k == 1:
        # Random Rotation
        theta = np.random.uniform(-ROT_MAX, ROT_MAX)
        for i in range (VIDEO_LENGTH):
            video[i] = random_rotation(video[i], (i*theta)/VIDEO_LENGTH, row_axis=0,
                            col_axis=1, channel_axis=2, fill_mode="nearest")
            video[i] = (video[i].astype(np.float32) - 127.5) / 127.5

    elif k == 2:
        # Random shift
        h, w = video.shape[1], video.shape[2]
        tx = np.random.uniform(-SFT_V_MAX, SFT_V_MAX) * h
        ty = np.random.uniform(-SFT_H_MAX, SFT_H_MAX) * w
        for i in range(VIDEO_LENGTH):
            video[i] = random_shift(video[i], tx, ty, row_axis=0,
                                    col_axis=1, channel_axis=2, fill_mode="nearest")
            video[i] = (video[i].astype(np.float32) - 127.5) / 127.5

    # elif k == 3:
    #     # Random zoom
    #     z = np.random.uniform(0, ZOOM_MAX, 1)
    #     for i in range(VIDEO_LENGTH):
    #         video[i] = random_zoom(video[i], ((i * z) / VIDEO_LENGTH), ((i * z) / VIDEO_LENGTH),
    #                                    row_axis=0, col_axis=1, channel_axis=2, fill_mode="nearest")

    elif k == 3:
        # Horizontal Flip
        for i in range(VIDEO_LENGTH):
            video[i] = flip_axis(video[i], axis=1)
            video[i] = (video[i].astype(np.float32) - 127.5) / 127.5

    else:
        # Vary brightness
        u = np.random.uniform(BRIGHT_RANGE_L, BRIGHT_RANGE_H)
        for i in range(VIDEO_LENGTH):
            video[i] = random_brightness(video[i], u)
            video[i] = (video[i].astype(np.float32) - 127.5) / 127.5

    return video


def load_X_y_RAM(videos_list, index, frames, ped_action_cats):
    if RAM_DECIMATE:
        X = []
        y = []
        for i in range(BATCH_SIZE):
            # start_index = videos_list[(index*BATCH_SIZE + i), 0]
            # end_index = videos_list[(index*BATCH_SIZE + i), -1]
            # X.append(frames[start_index:end_index+1])
            video = np.take(frames, videos_list[(index*BATCH_SIZE + i)], axis=0)
            video = random_augmentation(video)
            X.append(video)
            if (len(ped_action_cats) != 0):
                # y.append(ped_action_cats[start_index:end_index+1])
                y.append(np.take(ped_action_cats, videos_list[(index*BATCH_SIZE + i)], axis=0))

        X = np.asarray(X)
        y = np.asarray(y)
        return X, y
    else:
        print ("RAM usage flag not set. Are you sure you want to do this?")
        exit(0)


def load_X_y(videos_list, index, data_dir, ped_action_cats):
    X = np.zeros((BATCH_SIZE, VIDEO_LENGTH,) + IMG_SIZE)
    y = []
    for i in range(BATCH_SIZE):
        y_per_vid = []
        for j in range(VIDEO_LENGTH):
            frame_number = (videos_list[(index*BATCH_SIZE + i), j])
            filename = "frame_" + str(frame_number) + ".png"
            im_file = os.path.join(data_dir, filename)
            try:
                frame = cv2.imread(im_file, cv2.IMREAD_COLOR)
                # frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_LANCZOS4)
                X[i, j] = (frame.astype(np.float32) - 127.5) / 127.5
            except AttributeError as e:
                print (im_file)
                print (e)
            if (len(ped_action_cats) != 0):
                try:
                    y_per_vid.append(ped_action_cats[frame_number - 1])
                except IndexError as e:
                    print(frame_number)
                    print(e)
        if (len(ped_action_cats) != 0):
            y.append(y_per_vid)
    return X, np.asarray(y)


def map_to_simple(ped_action):
    if (ped_action == 0):
        return 0
    elif (ped_action == 1):
        return 1
    elif (ped_action == 2):
        return 1
    elif (ped_action == 5):
        return 2
    elif (ped_action == 6):
        return 2
    elif (ped_action == 7):
        return 2
    elif (ped_action == 8):
        return 0
    elif (ped_action == 9):
        return 1
    elif (ped_action == 12):
        return 3
    elif (ped_action == 13):
        return 4
    else:
        print ("Irrelevant ped_action found. Exiting.")
        print (ped_action)
        exit(0)


def get_action_classes(action_labels):
    # Load labels into per frame numerical indices from the action set
    print("Loading annotations.")

    ped_action_classes = []
    count = [0] * len(simple_ped_set)
    for i in range(len(action_labels)):
        action_dict = dict(ele.split(':') for ele in action_labels[i].split(', ')[2:])
        # Settle pedestrian classes
        a_clean = []
        for key, value in action_dict.iteritems():
            if 'pedestrian' in key:
                if ',' in value:
                    splits = value.split(',')
                    for k in range(len(splits)):
                        a_clean.append(splits[k])
                else:
                    a_clean.append(value)

        if len(a_clean) == 0:
            a_clean = ['no ped']

        ped_actions_per_frame = list(set([a.lower() for a in a_clean]))
        simple_ped_actions_per_frame = []
        encoded_ped_action = np.zeros(shape=(len(simple_ped_set)), dtype=np.float32)
        for action in ped_actions_per_frame:
            # Get ped action number and map it to simple set
            if action.lower() not in ped_actions:
                print ("Unknown action in labels. Exiting.")
                print (action)
                exit(0)
            if action.lower() == 'standing':
                ped_action = simple_ped_set.index('standing')
                simple_ped_actions_per_frame.append(ped_action)
            if action.lower() == 'crossing':
                ped_action = simple_ped_set.index('crossing')
                simple_ped_actions_per_frame.append(ped_action)
            if action.lower() == 'no ped':
                ped_action = simple_ped_set.index('no ped')
                simple_ped_actions_per_frame.append(ped_action)

            # ped_action = ped_actions.index(action)
            # if ((ped_action == ped_actions.index('nod')) or
            #     (ped_action == ped_actions.index('looking')) or
            #     (ped_action == ped_actions.index('nod')) or
            #     (ped_action == ped_actions.index('handwave'))):
            #     continue
            # else:
            #     ped_action = map_to_simple(ped_action)
            #     simple_ped_actions_per_frame.append(ped_action)

        # if 5 in simple_ped_action_per_frame:
        #     action = 5
        # if 6 in simple_ped_action_per_frame:
        #     action = 6
        # if 1 in simple_ped_action_per_frame:
        #     action = 1
        # if 4 in simple_ped_action_per_frame:
        #     action = 4

        # if 2 in simple_ped_actions_per_frame:
        #     act = 2
        # if 0 in simple_ped_actions_per_frame:
        #     act = 0
        # if 1 in simple_ped_actions_per_frame:
        #     act = 1
        #
        # encoded_ped_action = to_categorical(act, len(simple_ped_set))
        # count[act] = count[act] + 1

        for action in simple_ped_actions_per_frame:
            count[action] = count[action] + 1
            # Add all unique categorical one-hot vectors
            encoded_ped_action = encoded_ped_action + to_categorical(action, len(simple_ped_set))

        # if (sum(encoded_ped_action) == 0):
        #     print (ped_actions_per_frame)
        #     print (encoded_ped_action)

        # if (sum(encoded_ped_action) > 1):
        #     print (simple_ped_action_per_frame)
        #     print (a_clean)
        ped_action_classes.append(encoded_ped_action.T)

    ped_action_classes = np.asarray(ped_action_classes)
    ped_action_classes = np.reshape(ped_action_classes, newshape=(ped_action_classes.shape[0:2]))
    return ped_action_classes, count


def remove_zero_classes(videos_list, simple_ped_actions_per_frame):
    r_indices = []
    for i in range(len(videos_list)):
        # if (len(list(simple_ped_actions_per_frame[videos_list[i, CLASS_TARGET_INDEX]])) == 0):
        if sum(simple_ped_actions_per_frame[videos_list[i, CLASS_TARGET_INDEX]]) == 0:
            r_indices.append(i)

    for i in sorted(r_indices, reverse=True):
        videos_list = np.delete(videos_list, i, axis=0)

    return videos_list


def load_to_RAM(frames_source):
    frames = np.zeros(shape=((len(frames_source),) + IMG_SIZE))

    j = 1
    for i in range(1, len(frames_source)):
        filename = "frame_" + str(j) + ".png"
        im_file = os.path.join(DATA_DIR, filename)
        try:
            frame = cv2.imread(im_file, cv2.IMREAD_COLOR)
            # frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_LANCZOS4)
            frames[i] = frame.astype(np.float32)
            # frames[i] = (frame.astype(np.float32) - 127.5) / 127.5
            j = j + 1
        except AttributeError as e:
            print(im_file)
            print(e)

    return frames


def get_video_lists(frames_source, stride, frame_skip=0):
    # Build video progressions
    videos_list = []
    start_frame_index = 1
    end_frame_index = ((frame_skip + 1) * VIDEO_LENGTH) + 1 - frame_skip
    while (end_frame_index <= len(frames_source)):
        frame_list = frames_source[start_frame_index:end_frame_index]
        if (len(set(frame_list)) == 1):
            videos_list.append(range(start_frame_index, end_frame_index, frame_skip+1))
            start_frame_index = start_frame_index + stride
            end_frame_index = end_frame_index + stride
        else:
            start_frame_index = end_frame_index - 1
            end_frame_index = start_frame_index + (frame_skip+1)*VIDEO_LENGTH -frame_skip

    videos_list = np.asarray(videos_list, dtype=np.int32)

    return np.asarray(videos_list)


def get_classwise_data(videos_list, ped_action_labels):
    classwise_videos_list = [[] for _ in range(len(simple_ped_set))]
    count = [0] * len(simple_ped_set)
    for i in range(len(videos_list)):
        labels = np.where(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]] == 1)
        for j in labels[0]:
            count[j] += 1
            classwise_videos_list[j].append(np.asarray(videos_list[i]))

    print('Before subsampling')
    print(str(count))

    return classwise_videos_list, count

def prob_subsample(classwise_videos_list, count):
    train_videos_list = []
    sample_size = min(count)

    for i in range(len(classwise_videos_list)):
        indices = np.random.choice(count[i], sample_size, replace=False)
        videos_list = np.asarray(np.take(classwise_videos_list[i], indices, axis=0))
        train_videos_list.extend(np.asarray(videos_list))

    train_videos_list = np.random.permutation(train_videos_list)

    return np.asarray(train_videos_list)


def subsample_videos(videos_list, ped_action_labels):
    print (videos_list.shape)
    AP_MAX = 3
    CR_MAX = 10
    ST_MAX = 10
    NP_MAX = 3

    ap_count = 0
    cr_count = 0
    st_count = 0
    np_count = 0

    r_indices = []

    classwise_videos_list, count = get_classwise_data(videos_list, ped_action_labels)
    videos_list = prob_subsample(classwise_videos_list, count)

    exit(0)


    for i in range(len(videos_list)):
        # Approaching count
        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 1):
            ap_count = ap_count + 1
            if (ap_count < AP_MAX):
                r_indices.append(i)
            else:
                ap_count = 0

        # Crossing count
        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 2):
            cr_count = cr_count + 1
            if (cr_count < CR_MAX):
                r_indices.append(i)
            else:
                cr_count = 0

        # Stopped count
        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 3):
            st_count = st_count + 1
            if (st_count < ST_MAX):
                r_indices.append(i)
            else:
                st_count = 0

        # No ped count
        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 6):
            np_count = np_count + 1
            if (np_count < NP_MAX):
                r_indices.append(i)
            else:
                np_count = 0

        # if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) ==
        #         list(ped_action_labels[videos_list[i, 8]]).index(1)):
        #     r_indices.append(i)

    for i in sorted(r_indices, reverse=True):
        videos_list = np.delete(videos_list, i, axis=0)

    count = [0] * len(simple_ped_set)
    for i in range(len(videos_list)):
        count[list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1)] = \
            count[list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1)] + 1

    print ('After subsampling')
    print (str(count))

    return videos_list


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_confusion_matrix(y_true, y_pred):
    cnf_matrix = confusion_matrix(y_true, np.round(y_pred), labels=simple_ped_set)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=simple_ped_set, normalize=True,
                      title='Normalized confusion matrix')
    plt.show()


def get_sklearn_metrics(y_true, y_pred, avg=None):
    # y_true_labels = []
    # y_pred_labels = []
    # for i in range(y_true.shape[0]):
    #     y_true_labels.append(np.argmax(y_true[i]))
    #     y_pred_labels.append(np.argmax(y_pred[i]))

    # return precision_recall_fscore_support(y_true_labels, y_pred_labels, average=avg)
    return precision_recall_fscore_support(y_true, np.round(y_pred), average=avg)


def get_classification_report(y_true, y_pred):
    return classification_report(y_true, np.round(y_pred), target_names=simple_ped_set)


def train(BATCH_SIZE, ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS):
    print("Loading data definitions.")

    frames_source = hkl.load(os.path.join(DATA_DIR, 'sources_train_208.hkl'))
    videos_list_1 = get_video_lists(frames_source=frames_source, stride=1, frame_skip=0)
    videos_list_2 = get_video_lists(frames_source=frames_source, stride=1, frame_skip=1)
    # videos_list_3 = get_video_lists(frames_source=frames_source, stride=1, frame_skip=2)
    videos_list = np.concatenate((videos_list_1, videos_list_2), axis=0)

    # Load actions from annotations
    action_labels = hkl.load(os.path.join(DATA_DIR, 'annotations_train_208.hkl'))
    ped_action_classes, ped_class_count = get_action_classes(action_labels=action_labels)
    print("Training Stats: " + str(ped_class_count))

    # videos_list = remove_zero_classes(videos_list, ped_action_classes)
    classwise_videos_list, count = get_classwise_data(videos_list, ped_action_classes)
    videos_list = prob_subsample(classwise_videos_list, count)

    if RAM_DECIMATE:
        frames = load_to_RAM(frames_source=frames_source)

    # if SHUFFLE:
    #     # Shuffle images to aid generalization
    #     videos_list = np.random.permutation(videos_list)

    # Setup test
    test_frames_source = hkl.load(os.path.join(TEST_DATA_DIR, 'sources_test_208.hkl'))
    test_videos_list = get_video_lists(frames_source=test_frames_source, stride=4)
    # Load test action annotations
    test_action_labels = hkl.load(os.path.join(TEST_DATA_DIR, 'annotations_test_208.hkl'))
    test_ped_action_classes, test_ped_class_count = get_action_classes(test_action_labels)
    test_videos_list = remove_zero_classes(test_videos_list, test_ped_action_classes)
    print("Test Stats: " + str(test_ped_class_count))

    # Build the Spatio-temporal Autoencoder
    print ("Creating models.")
    # Build stacked classifier
    classifier = pretrained_c3d()
    # classifier = c3d_scratch()
    classifier.compile(loss="binary_crossentropy",
                       optimizer=OPTIM_C,
                       # metrics=[metric_precision, metric_recall, metric_mpca, 'accuracy'])
                       metrics=['accuracy'])

    run_utilities(classifier, CLA_WEIGHTS)

    n_videos = videos_list.shape[0]
    n_test_videos = test_videos_list.shape[0]
    NB_ITERATIONS = int(n_videos/BATCH_SIZE)
    # NB_ITERATIONS = 5
    NB_TEST_ITERATIONS = int(n_test_videos/BATCH_SIZE)
    # NB_TEST_ITERATIONS = 5

    # Setup TensorBoard Callback
    TC_cla = tb_callback.TensorBoard(log_dir=TF_LOG_CLA_DIR, histogram_freq=0, write_graph=False, write_images=False)
    LRS_clas = lrs_callback.LearningRateScheduler(schedule=schedule)
    LRS_clas.set_model(classifier)


    print ("Beginning Training.")
    # Begin Training
    # Train Classifier
    if CLASSIFIER:
        print("Training Classifier...")
        for epoch in range(NB_EPOCHS_CLASS):
            print("\n\nEpoch ", epoch)
            c_loss = []
            test_c_loss = []

            # # Set learning rate every epoch
            LRS_clas.on_epoch_begin(epoch=epoch)
            lr = K.get_value(classifier.optimizer.lr)
            print("Learning rate: " + str(lr))
            print("c_loss_metrics: " + str(classifier.metrics_names))

            y_train_pred = []
            y_train_true = []
            for index in range(NB_ITERATIONS):
                # Train Autoencoder
                if RAM_DECIMATE:
                    videos_list = prob_subsample(classwise_videos_list, count)
                    X, y = load_X_y_RAM(videos_list, index, frames, ped_action_classes)
                else:
                    videos_list = prob_subsample(classwise_videos_list, count)
                    X, y = load_X_y(videos_list, index, DATA_DIR, ped_action_classes)

                X_train = X
                y_true_class = y[:, CLASS_TARGET_INDEX]

                c_loss.append(classifier.train_on_batch(X_train, y_true_class))
                y_train_true.extend(y_true_class)
                y_train_pred.extend(classifier.predict(X_train, verbose=0))

                arrow = int(index / (NB_ITERATIONS / 30))
                stdout.write("\rIter: " + str(index) + "/" + str(NB_ITERATIONS - 1) + "  " +
                             "c_loss: " + str([ c_loss[len(c_loss) - 1][j]  for j in [0, 1]]) + "  " +
                             "\t    [" + "{0}>".format("=" * (arrow)))
                stdout.flush()

            if SAVE_GENERATED_IMAGES:
                # Save generated images to file
                ped_pred_class = classifier.predict(X_train, verbose=0)
                # pred_seq = arrange_images(np.concatenate((X_train, predicted_images), axis=1))
                pred_seq = arrange_images(X_train)
                pred_seq = pred_seq * 127.5 + 127.5

                font = cv2.FONT_HERSHEY_SIMPLEX
                y_orig_classes = y
                # Add labels as text to the image

                for k in range(BATCH_SIZE):
                    for j in range(int(VIDEO_LENGTH )):
                        class_num_past = np.argmax(y_orig_classes[k, j])
                        class_num_y = np.argmax(ped_pred_class[k])
                        # label_true = simple_ped_set[class_num_past]
                        # label_pred = simple_ped_set[class_num_y]

                        label_true = str(y_orig_classes[k, j])
                        label_pred = str([round(float(i), 2) for i in ped_pred_class[k]])

                        # if (y_orig_classes[k, j] > 0.5):
                        #     label_true = "crossing"
                        # else:
                        #     label_true = "not crossing"
                        #
                        # if (ped_pred_class[k] > 0.5):
                        #     label_pred = "crossing"
                        # else:
                        #     label_pred = "not crossing"

                        cv2.putText(pred_seq, 'truth: ' + label_true,
                                    (2 + j * (208), 94 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        cv2.putText(pred_seq, label_pred,
                                    (2 + j * (208), 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)

                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_cla_pred.png"), pred_seq)

            # Run over test data
            print('')
            y_test_pred = []
            y_test_true = []
            for index in range(NB_TEST_ITERATIONS):
                X, y = load_X_y(test_videos_list, index, TEST_DATA_DIR, test_ped_action_classes)
                X_test = X
                y_true_class = y[:, CLASS_TARGET_INDEX]

                test_c_loss.append(classifier.test_on_batch(X_test, y_true_class))
                y_test_true.extend(y_true_class)
                y_test_pred.extend(classifier.predict(X_test, verbose=0))

                arrow = int(index / (NB_TEST_ITERATIONS / 40))
                stdout.write("\rIter: " + str(index) + "/" + str(NB_TEST_ITERATIONS - 1) + "  " +
                             "test_c_loss: " +  str([ test_c_loss[len(test_c_loss) - 1][j]  for j in [0, 1]]))
                stdout.flush()

            if SAVE_GENERATED_IMAGES:
                # Save generated images to file
                test_ped_pred_class = classifier.predict(X_test, verbose=0)
                # pred_seq = arrange_images(np.concatenate((X_train, predicted_images), axis=1))
                pred_seq = arrange_images(X_test)
                pred_seq = pred_seq * 127.5 + 127.5

                font = cv2.FONT_HERSHEY_SIMPLEX
                y_orig_classes = y
                # Add labels as text to the image

                for k in range(BATCH_SIZE):
                    for j in range(int(VIDEO_LENGTH)):
                        class_num_past = np.argmax(y_orig_classes[k, j])
                        class_num_y = np.argmax(test_ped_pred_class[k])
                        # label_true = simple_ped_set[class_num_past]
                        # label_pred = simple_ped_set[class_num_y]
                        label_true = str(y_orig_classes[k, j])
                        label_pred = str([round(float(i), 2) for i in ped_pred_class[k]])

                        #
                        # if (y_orig_classes[k, j] > 0.5):
                        #     label_true = "crossing"
                        # else:
                        #     label_true = "not crossing"
                        #
                        # if (test_ped_pred_class[k] > 0.5):
                        #     label_pred = "crossing"
                        # else:
                        #     label_pred = "not crossing"

                        cv2.putText(pred_seq, 'truth: ' + label_true,
                                    (2 + j * (208), 94 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        cv2.putText(pred_seq, label_pred,
                                    (2 + j * (208), 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)

                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_cla_test_pred.png"), pred_seq)

            # then after each epoch
            avg_c_loss = np.mean(np.asarray(c_loss, dtype=np.float32), axis=0)
            avg_test_c_loss = np.mean(np.asarray(test_c_loss, dtype=np.float32), axis=0)

            print (np.asarray(y_train_true))
            print (np.asarray(y_train_pred))

            train_prec, train_rec, train_fbeta, train_support = get_sklearn_metrics(np.asarray(y_train_true),
                                                                                    np.asarray(y_train_pred),
                                                                                    avg=None)
            test_prec, test_rec, test_fbeta, test_support = get_sklearn_metrics(np.asarray(y_test_true),
                                                                                np.asarray(y_test_pred),
                                                                                avg=None)

            loss_values = np.asarray(avg_c_loss.tolist() + train_prec.tolist() +
                                     train_rec.tolist() +
                                     avg_test_c_loss.tolist() + test_prec.tolist() +
                                     test_rec.tolist(), dtype=np.float32)
            precs = ['prec_' + action for action in simple_ped_set]
            recs = ['rec_' + action for action in simple_ped_set]
            fbeta = ['fbeta_' + action for action in simple_ped_set]
            c_loss_keys = ['c_' + metric for metric in classifier.metrics_names+precs+recs]
            test_c_loss_keys = ['c_test_' + metric for metric in classifier.metrics_names+precs+recs]

            loss_keys = c_loss_keys + test_c_loss_keys
            logs = dict(zip(loss_keys, loss_values))

            TC_cla.on_epoch_end(epoch, logs)

            # Log the losses
            with open(os.path.join(LOG_DIR, 'losses_cla.json'), 'a') as log_file:
                log_file.write("{\"epoch\":%d, %s;\n" % (epoch, logs))

            print("\nAvg c_loss: " + str(avg_c_loss) +
                  " Avg test_c_loss: " + str(avg_test_c_loss))

            print ("Training Precision per class:" + str(train_prec))
            print ("Test Precision per class:" + str(test_prec))
            print ("Training Recall per class:" + str(train_rec))
            print ("Test Recall per class:" + str(test_rec))

            prec, recall, fbeta, support = get_sklearn_metrics(np.asarray(y_train_true),
                                                               np.asarray(y_train_pred),
                                                               avg='weighted')
            print ("Train Prec: %.2f, Recall: %.2f, Fbeta: %.2f" %(prec, recall, fbeta))
            prec, recall, fbeta, support = get_sklearn_metrics(np.asarray(y_test_true),
                                                               np.asarray(y_test_pred),
                                                               avg='weighted')
            print("Test Prec: %.2f, Recall: %.2f, Fbeta: %.2f" % (prec, recall, fbeta))

            # Save model weights per epoch to file
            # encoder.save_weights(os.path.join(CHECKPOINT_DIR, 'encoder_cla_epoch_' + str(epoch) + '.h5'), True)
            # decoder.save_weights(os.path.join(CHECKPOINT_DIR, 'decoder_cla_epoch_' + str(epoch) + '.h5'), True)
            classifier.save_weights(os.path.join(CHECKPOINT_DIR, 'classifier_cla_epoch_' + str(epoch) + '.h5'),
                                    True)

            # get_confusion_matrix(y_train_true, y_train_pred)
            # get_confusion_matrix(y_test_true, y_test_pred)

        print (get_classification_report(np.asarray(y_train_true), np.asarray(y_train_pred)))
        print (get_classification_report(np.asarray(y_test_true), np.asarray(y_test_pred)))


def test(ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS):

    # Create models
    print ("Creating models.")
    classifier = pretrained_c3d()
    classifier.compile(loss="binary_crossentropy",
                       optimizer=OPTIM_C,
                       metrics=[metric_precision, metric_recall, metric_mpca, 'accuracy'])

    run_utilities(classifier, CLA_WEIGHTS)

    # Setup test
    test_frames_source = hkl.load(os.path.join(TEST_DATA_DIR, 'sources_test_208.hkl'))
    test_videos_list = get_video_lists(frames_source=test_frames_source, stride=8)
    # Load test action annotations
    test_action_labels = hkl.load(os.path.join(TEST_DATA_DIR, 'annotations_test_208.hkl'))
    test_driver_action_classes, test_ped_action_classes, test_ped_class_count = get_action_classes(test_action_labels)
    print("Test Stats: " + str(test_ped_class_count))

    n_test_videos = test_videos_list.shape[0]

    # Test model by making predictions
    test_c_loss = []
    y_true_total = []
    y_pred_total = []
    NB_TEST_ITERATIONS = int(n_test_videos / BATCH_SIZE)
    for index in range(NB_TEST_ITERATIONS):
        X, y1, y2 = load_X_y(test_videos_list, index, TEST_DATA_DIR, [], test_ped_action_classes)
        X_test = X
        y2_true_class = y2[:, CLASS_TARGET_INDEX]

        test_c_loss.append(classifier.test_on_batch(X_test, y2_true_class))
        y_true_total.append(K.cast(y2_true_class, dtype='int32'))
        y_pred_total.append(K.cast(K.round(classifier.predict(X_test)), 'int32'))

        arrow = int(index / (NB_TEST_ITERATIONS / 40))
        stdout.write("\rIter: " + str(index) + "/" + str(NB_TEST_ITERATIONS - 1) + "  " +
                     "test_c_loss: " + str([test_c_loss[len(test_c_loss) - 1][j] for j in [0, 1, 2, 3, 4]]))
        stdout.flush()

        # Save generated images to file
        ped_pred_class = classifier.predict(X_test, verbose=0)
        orig_image = arrange_images(X_test)
        orig_image = orig_image * 127.5 + 127.5
        pred_image = np.copy(orig_image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        y2_orig_classes = y2
        # Add labels as text to the image
        for k in range(BATCH_SIZE):
            for j in range(VIDEO_LENGTH):
                class_num_past_y2 = np.argmax(y2_orig_classes[k, j])
                cv2.putText(orig_image, "Ped: " + simple_ped_set[class_num_past_y2],
                            (2 + j * (112), 104 + k * 112), font, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
        cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(index) +
                                 "_cla_test_orig.png"), orig_image)

        # Add labels as text to the image
        for k in range(BATCH_SIZE):
            class_num_y2 = np.argmax(ped_pred_class[k])
            cv2.putText(pred_image, "Ped: " + simple_ped_set[class_num_y2],
                        (2, 104 + k * 112), font, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
        cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(index) + "_cla_test_pred.png"),
                    pred_image)

    # then after each epoch/iteration
    avg_test_c_loss = np.mean(np.asarray(test_c_loss, dtype=np.float32), axis=0)

    loss_values = np.asarray(avg_test_c_loss.tolist(), dtype=np.float32)
    test_c_loss_keys = ['c_test_' + metric for metric in classifier.metrics_names]

    loss_keys = test_c_loss_keys
    logs = dict(zip(loss_keys, loss_values))

    # Log the losses
    with open(os.path.join(LOG_DIR, 'losses_cla.json'), 'a') as log_file:
        log_file.write("{\"Iter\":%d, %s;\n" % (index, logs))

    print("\nAvg test_c_loss: " + str(avg_test_c_loss))

    np.save(os.path.join(CLA_GEN_IMAGES_DIR, 'y_true_total.npy'), np.asarray(y_true_total))
    np.save(os.path.join(CLA_GEN_IMAGES_DIR, 'y_pred_total.npy'), np.asarray(y_pred_total))

    precision, recall, f1 = score(np.asarray(y_true_total), np.asarray(y_pred_total))

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(f1))




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--enc_weights", type=str, default="None")
    parser.add_argument("--dec_weights", type=str, default="None")
    parser.add_argument("--cla_weights", type=str, default="None")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size,
              ENC_WEIGHTS=args.enc_weights,
              DEC_WEIGHTS=args.dec_weights,
              CLA_WEIGHTS=args.cla_weights)

    if args.mode == "test":
        test(ENC_WEIGHTS=args.enc_weights,
             DEC_WEIGHTS=args.dec_weights,
             CLA_WEIGHTS=args.cla_weights)

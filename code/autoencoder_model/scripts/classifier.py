from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hickle as hkl
import numpy as np

np.random.seed(2 ** 10)
from keras import backend as K
K.set_image_dim_ordering('tf')
import tensorflow as tf
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
from custom_layers import AttnLossLayer
from keras.models import model_from_json
from keras.metrics import top_k_categorical_accuracy

from experience_memory import ExperienceMemory
from config_classifier import *
from sys import stdout

import tb_callback
import lrs_callback
import argparse
import random
import math
import cv2
import os


def encoder_model():
    model = Sequential()

    # 10x128x128
    model.add(Conv3D(filters=128,
                     strides=(1, 4, 4),
                     kernel_size=(3, 11, 11),
                     padding='same',
                     input_shape=(int(VIDEO_LENGTH/2), 128, 128, 3)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
    model.add(TimeDistributed(Dropout(0.5)))

    # 10x32x32
    model.add(Conv3D(filters=64,
                     strides=(1, 2, 2),
                     kernel_size=(3, 5, 5),
                     padding='same'))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
    model.add(TimeDistributed(Dropout(0.5)))

    # 10x16x16
    model.add(Conv3D(filters=64,
                     strides=(1, 1, 1),
                     kernel_size=(3, 3, 3),
                     padding='same'))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
    model.add(TimeDistributed(Dropout(0.5)))

    return model


def decoder_model():
    inputs = Input(shape=(int(VIDEO_LENGTH/2), 16, 16, 64))

    # 10x16x16
    convlstm_1 = ConvLSTM2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(inputs)
    x = TimeDistributed(BatchNormalization())(convlstm_1)
    x = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_1 = TimeDistributed(Dropout(0.5))(x)

    flat_1 = TimeDistributed(Flatten())(out_1)
    aclstm_1 = GRU(units=16 * 16,
                   recurrent_dropout=0.2,
                   return_sequences=True)(flat_1)
    x = TimeDistributed(BatchNormalization())(aclstm_1)
    dense_1 = TimeDistributed(Dense(units=16 * 16, activation='softmax'))(x)
    a1_reshape = Reshape(target_shape=(int(VIDEO_LENGTH/2), 16, 16, 1))(dense_1)
    a1 = AttnLossLayer()(a1_reshape)
    dot_1 = multiply([out_1, a1])

    convlstm_2 = ConvLSTM2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(dot_1)
    x = TimeDistributed(BatchNormalization())(convlstm_2)
    h_2 = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_2 = UpSampling3D(size=(1, 2, 2))(h_2)

    skip_upsamp_1 = UpSampling3D(size=(1, 2, 2))(dot_1)
    res_1 = concatenate([out_2, skip_upsamp_1])

    # 10x32x32
    convlstm_3 = ConvLSTM2D(filters=128,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(res_1)
    x = TimeDistributed(BatchNormalization())(convlstm_3)
    h_3 = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_3 = UpSampling3D(size=(1, 2, 2))(h_3)

    skip_upsamp_2 = UpSampling3D(size=(1, 2, 2))(out_2)
    res_2 = concatenate([out_3, skip_upsamp_2])

    # 10x64x64
    convlstm_4 = ConvLSTM2D(filters=32,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(res_2)
    x = TimeDistributed(BatchNormalization())(convlstm_4)
    h_4 = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_4 = UpSampling3D(size=(1, 2, 2))(h_4)

    # 10x128x128
    convlstm_5 = ConvLSTM2D(filters=3,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(out_4)
    predictions = TimeDistributed(Activation('tanh'))(convlstm_5)

    model = Model(inputs=inputs, outputs=predictions)

    return model



def process_prec3d():
    json_file = open(PRETRAINED_C3D, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    model.load_weights(PRETRAINED_C3D_WEIGHTS)
    print("Loaded model from disk")
    for layer in model.layers[:13]:
        layer.trainable = RETRAIN_CLASSIFIER

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

    # i = 0
    # for layer in model.layers:
    #     print(layer, i)
    #     i = i + 1
    #
    # exit(0)

    return model


def pretrained_c3d():
    c3d = process_prec3d()

    inputs = Input(shape=(16, 128, 128, 3))
    resized = TimeDistributed(Lambda(lambda image: tf.image.resize_images(image, (112, 112))))(inputs)

    c3d_out = c3d(resized)
    dense = Dense(units=1024, activation='tanh')(c3d_out)
    x = BatchNormalization()(dense)
    x = Dropout(0.5)(x)
    x = Dense(units=512, activation='tanh')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    actions = Dense(units=len(simple_ped_set), activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=actions)

    # i=0
    # for layer in model.layers:
    #     print (layer, i)
    #     i = i+1
    # exit(0)

    return model


def load_to_RAM(frames_source):
    frames = np.zeros(shape=((len(frames_source),) + IMG_SIZE))
    print ('Decimating RAM!')
    j = 1
    for i in range(1, len(frames_source)):
        filename = "frame_" + str(j) + ".png"
        im_file = os.path.join(DATA_DIR, filename)
        try:
            frame = cv2.imread(im_file, cv2.IMREAD_COLOR)
            # frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_CUBIC)
            frames[i] = (frame.astype(np.float32) - 127.5) / 127.5
            j = j + 1
        except AttributeError as e:
            print(im_file)
            print(e)

    return frames


def set_trainability(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def autoencoder_model(encoder, decoder):
    model = Sequential()
    model.add(encoder)
    model.add(decoder)
    return model


def stacked_classifier_model(encoder, decoder, classifier):
    input = Input(shape=(16, 128, 128, 3))
    set_trainability(encoder, False)
    z = encoder(input)
    set_trainability(decoder, False)
    future = decoder(z)
    actions = classifier(future)

    model = Model(inputs=input, outputs=actions)

    return model


def arrange_images(video_stack):
    n_frames = video_stack.shape[0] * video_stack.shape[1]
    frames = np.zeros((n_frames,) + video_stack.shape[2:], dtype=video_stack.dtype)

    frame_index = 0
    for i in range(video_stack.shape[0]):
        for j in range(video_stack.shape[1]):
            frames[frame_index] = video_stack[i, j]
            frame_index += 1

    img_height = video_stack.shape[3]
    img_width = video_stack.shape[2]
    width = img_width * video_stack.shape[1]
    height = img_height * BATCH_SIZE
    shape = frames.shape[1:]
    image = np.zeros((height, width, shape[2]), dtype=video_stack.dtype)
    frame_number = 0
    for i in range(BATCH_SIZE):
        for j in range(video_stack.shape[1]):
            image[(i * img_height):((i + 1) * img_height), (j * img_width):((j + 1) * img_width)] = frames[frame_number]
            frame_number = frame_number + 1

    return image


def load_weights(weights_file, model):
    model.load_weights(weights_file)


def run_utilities(encoder, decoder, autoencoder, classifier, ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS):
    if PRINT_MODEL_SUMMARY:
        print ("Encoder:")
        print (encoder.summary())
        print ("Decoder:")
        print (decoder.summary())
        print ("Autoencoder:")
        print (autoencoder.summary())
        if CLASSIFIER:
            print("Classifier:")
            print (classifier.summary())

    # Save model to file
    if SAVE_MODEL:
        print ("Saving models to file.")
        model_json = encoder.to_json()
        with open(os.path.join(MODEL_DIR, "encoder.json"), "w") as json_file:
            json_file.write(model_json)

        model_json = decoder.to_json()
        with open(os.path.join(MODEL_DIR, "decoder.json"), "w") as json_file:
            json_file.write(model_json)

        model_json = autoencoder.to_json()
        with open(os.path.join(MODEL_DIR, "autoencoder.json"), "w") as json_file:
            json_file.write(model_json)

        if CLASSIFIER:
            model_json = classifier.to_json()
            with open(os.path.join(MODEL_DIR, "classifier.json"), "w") as json_file:
                json_file.write(model_json)

        if PLOT_MODEL:
            plot_model(encoder, to_file=os.path.join(MODEL_DIR, 'encoder.png'), show_shapes=True)
            plot_model(decoder, to_file=os.path.join(MODEL_DIR, 'decoder.png'), show_shapes=True)
            plot_model(autoencoder, to_file=os.path.join(MODEL_DIR, 'autoencoder.png'), show_shapes=True)
            if CLASSIFIER:
                plot_model(classifier, to_file=os.path.join(MODEL_DIR, 'classifier.png'), show_shapes=True)

    if ENC_WEIGHTS != "None":
        print ("Pre-loading encoder with weights.")
        load_weights(ENC_WEIGHTS, encoder)
    if DEC_WEIGHTS != "None":
        print ("Pre-loading decoder with weights.")
        load_weights(DEC_WEIGHTS, decoder)
    if CLASSIFIER:
        if CLA_WEIGHTS != "None":
            print("Pre-loading classifier with weights.")
            load_weights(CLA_WEIGHTS, classifier)


def load_X_y_RAM(videos_list, index, frames, ped_action_cats):
    if RAM_DECIMATE:
        X = []
        y = []
        for i in range(BATCH_SIZE):
            start_index = videos_list[(index*BATCH_SIZE + i), 0]
            end_index = videos_list[(index*BATCH_SIZE + i), -1]
            X.append(frames[start_index:end_index+1])
            if (len(ped_action_cats) != 0):
                y.append(ped_action_cats[start_index:end_index+1])

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


def load_to_RAM(frames_source):
    frames = np.zeros(shape=((len(frames_source),) + IMG_SIZE))

    j = 1
    for i in range(1, len(frames_source)):
        filename = "frame_" + str(j) + ".png"
        im_file = os.path.join(DATA_DIR, filename)
        try:
            frame = cv2.imread(im_file, cv2.IMREAD_COLOR)
            # frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_LANCZOS4)
            frames[i] = (frame.astype(np.float32) - 127.5) / 127.5
            j = j + 1
        except AttributeError as e:
            print(im_file)
            print(e)

    return frames


def map_to_simple(ped_action):
    if (ped_action <= 1):
        return 0
    elif (ped_action <= 3):
        return 1
    elif (ped_action <= 5):
        return 0
    elif (ped_action <= 7):
        return 2
    elif (ped_action == 8):
        return 3
    elif (ped_action == 9):
        return 0
    elif (ped_action <= 11):
        return 2
    else:
        return 4


def get_action_classes(action_labels):
    # Load labesl into categorical 1-hot vectors
    print("Loading annotations.")
    count = [0] * len(simple_ped_set)
    driver_action_class = []
    ped_action_class = []
    # action_class = []
    for i in range(len(action_labels)):
        action_dict = dict(ele.split(':') for ele in action_labels[i].split(', ')[2:])
        # Settle pedestrian classes
        # print (action_dict)
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
            a_clean = ['unknown']

        ped_action_per_frame = list(set(a_clean))
        # print(ped_action_per_frame)
        encoded_ped_action = np.zeros(shape=(len(simple_ped_set)), dtype=np.float32)
        # print (ped_action_per_frame)
        simple_ped_action_per_frame = []
        for action in ped_action_per_frame:
            # Get ped action number and map it to simple set
            ped_action = ped_actions.index(action)
            ped_action = map_to_simple(ped_action)
            simple_ped_action_per_frame.append(ped_action)


        simple_ped_action_per_frame = set(simple_ped_action_per_frame)

        if 7 in simple_ped_action_per_frame:
            action = 7
        if 6 in simple_ped_action_per_frame:
            action = 6
        if 2 in simple_ped_action_per_frame:
            action = 2
        if 0 in simple_ped_action_per_frame:
            action = 0
        if 1 in simple_ped_action_per_frame:
            action = 1
        if 5 in simple_ped_action_per_frame:
            action = 5
        if 3 in simple_ped_action_per_frame:
            action = 3
        if 4 in simple_ped_action_per_frame:
            action = 4

        encoded_ped_action = to_categorical(action, len(simple_ped_set))
        count[action] = count[action] + 1

        # for action in simple_ped_action_per_frame:
        #     count[action] = count[action] + 1
        #     # Add all unique categorical one-hot vectors
        #     encoded_ped_action = encoded_ped_action + to_categorical(action, len(simple_ped_set))

        if (sum(encoded_ped_action) == 0):
            print (simple_ped_action_per_frame)
            print (a_clean)
        if (sum(encoded_ped_action) > 1):
            print (simple_ped_action_per_frame)
            print (a_clean)
        ped_action_class.append(encoded_ped_action.T)
        # action_class.append((encoded_driver_action + encoded_ped_action).T)

    driver_action_class = np.asarray(driver_action_class)
    driver_action_class = np.reshape(driver_action_class, newshape=(driver_action_class.shape[0:2]))

    ped_action_class = np.asarray(ped_action_class)
    ped_action_class = np.reshape(ped_action_class, newshape=(ped_action_class.shape[0:2]))


    return ped_action_class, count


def subsample_videos(videos_list, ped_action_labels):
    print (videos_list.shape)
    CR_MAX = 22
    UK_MAX = 9
    ST_MAX = 6
    cr_count = 0
    uk_count = 0
    st_count = 0
    r_indices = []

    count = [0] * len(simple_ped_set)
    for i in range(len(videos_list)):
        # Crossing count
        # print (list(driver_action_labels[videos_list[i, 8]]).index(1))
        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 0):
            count[0] = count[0] + 1

        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 1):
            count[1] = count[1] + 1

        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 2):
            count[2] = count[2] + 1

        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 3):
            count[3] = count[3] + 1

        # Unknown count
        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 4):
            count[4] = count[4] + 1

    print('Before subsampling')
    print(str(count))

    for i in range(len(videos_list)):
        # Crossing count
        # print(list(driver_action_labels[videos_list[i, 8]]).index(1))
        # if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 0):
        #     cr_count = cr_count + 1
        #     if (cr_count < CR_MAX):
        #         r_indices.append(i)
        #     else:
        #         cr_count = 0
        #
        # # Stopped count
        # if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 1):
        #     st_count = st_count + 1
        #     if (st_count < ST_MAX):
        #         r_indices.append(i)
        #     else:
        #         st_count = 0
        #
        # # Unknown count
        # if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 4):
        #     uk_count = uk_count + 1
        #     if (uk_count < UK_MAX):
        #         r_indices.append(i)
        #     else:
        #         uk_count = 0

        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) ==
                list(ped_action_labels[videos_list[i, 8]]).index(1)):
            r_indices.append(i)


    for i in sorted(r_indices, reverse=True):
        videos_list = np.delete(videos_list, i, axis=0)

    count = [0] * len(simple_ped_set)
    for i in range(len(videos_list)):
        # Crossing count
        # print (list(driver_action_labels[videos_list[i, 8]]).index(1))
        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 0):
            count[0] = count[0] + 1

        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 1):
            count[1] = count[1] + 1

        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 2):
            count[2] = count[2] + 1

        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 3):
            count[3] = count[3] + 1

        # Unknown count
        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 4):
            count[4] = count[4] + 1

    print ('After subsampling')
    print (str(count))

    return videos_list


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


def top_2_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def get_video_lists(frames_source, stride):
    # Build video progressions
    videos_list = []
    start_frame_index = 1
    end_frame_index = VIDEO_LENGTH + 1
    while (end_frame_index <= len(frames_source)):
        frame_list = frames_source[start_frame_index:end_frame_index]
        if (len(set(frame_list)) == 1):
            videos_list.append(range(start_frame_index, end_frame_index))
            start_frame_index = start_frame_index + stride
            end_frame_index = end_frame_index + stride
        else:
            start_frame_index = end_frame_index - 1
            end_frame_index = start_frame_index + VIDEO_LENGTH

    videos_list = np.asarray(videos_list, dtype=np.int32)

    return np.asarray(videos_list)


def train(BATCH_SIZE, ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS):
    print("Loading data definitions.")

    frames_source = hkl.load(os.path.join(DATA_DIR, 'sources_train_128.hkl'))
    videos_list = get_video_lists(frames_source=frames_source, stride=1)
    # Load actions from annotations
    action_labels = hkl.load(os.path.join(DATA_DIR, 'annotations_train_128.hkl'))
    ped_action_classes, ped_class_count = get_action_classes(action_labels=action_labels)
    print("Training Stats: " + str(ped_class_count))

    if RAM_DECIMATE:
        frames = load_to_RAM(frames_source=frames_source)

    if SHUFFLE:
        # Shuffle images to aid generalization
        videos_list = np.random.permutation(videos_list)

    # Setup test
    test_frames_source = hkl.load(os.path.join(TEST_DATA_DIR, 'sources_test_128.hkl'))
    test_videos_list = get_video_lists(frames_source=test_frames_source, stride=8)
    # Load test action annotations
    test_action_labels = hkl.load(os.path.join(TEST_DATA_DIR, 'annotations_test_128.hkl'))
    test_ped_action_classes, test_ped_class_count = get_action_classes(test_action_labels)
    print("Test Stats: " + str(test_ped_class_count))

    videos_list = subsample_videos(videos_list=videos_list, ped_action_labels=ped_action_classes)

    # Build the Spatio-temporal Autoencoder
    print ("Creating models.")
    encoder = encoder_model()
    decoder = decoder_model()

    print (encoder.summary())
    print (decoder.summary())

    # Build attention layer output
    intermediate_decoder = Model(inputs=decoder.layers[0].input, outputs=decoder.layers[10].output)
    mask_gen = Sequential()
    mask_gen.add(encoder)
    mask_gen.add(intermediate_decoder)
    mask_gen.compile(loss='mean_absolute_error', optimizer=OPTIM_A)

    autoencoder = autoencoder_model(encoder, decoder)
    autoencoder.compile(loss="mean_absolute_error", optimizer=OPTIM_A)

        # Build stacked classifier
    if CLASSIFIER:
        classifier = pretrained_c3d()
        classifier.compile(loss="binary_crossentropy",
                           optimizer=OPTIM_C,
                           metrics=[metric_precision, metric_recall, metric_mpca, 'accuracy'])
        sclassifier = stacked_classifier_model(encoder, decoder, classifier)
        sclassifier.compile(loss="binary_crossentropy",
                            optimizer=OPTIM_C,
                            metrics=[metric_precision, metric_recall, metric_mpca, 'accuracy'])
        print (sclassifier.summary())

    if CLASSIFIER:
        run_utilities(encoder, decoder, autoencoder, classifier, ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS)
    else:
        run_utilities(encoder, decoder, autoencoder, 'classifier', ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS)

    n_videos = videos_list.shape[0]
    n_test_videos = test_videos_list.shape[0]
    NB_ITERATIONS = int(n_videos/BATCH_SIZE)
    # NB_ITERATIONS = 1
    NB_TEST_ITERATIONS = int(n_test_videos/BATCH_SIZE)
    # NB_TEST_ITERATIONS = 1

    # Setup TensorBoard Callback
    TC = tb_callback.TensorBoard(log_dir=TF_LOG_DIR, histogram_freq=0, write_graph=False, write_images=False)
    TC_cla = tb_callback.TensorBoard(log_dir=TF_LOG_CLA_DIR, histogram_freq=0, write_graph=False, write_images=False)
    LRS_auto = lrs_callback.LearningRateScheduler(schedule=auto_schedule)
    LRS_auto.set_model(autoencoder)
    if CLASSIFIER:
        LRS_clas = lrs_callback.LearningRateScheduler(schedule=clas_schedule)
        LRS_clas.set_model(sclassifier)


    print ("Beginning Training.")
    # Begin Training
    for epoch in range(NB_EPOCHS_AUTOENCODER):
        print("\n\nEpoch ", epoch)
        loss = []
        test_loss = []

        # Set learning rate every epoch
        LRS_auto.on_epoch_begin(epoch=epoch)
        lr = K.get_value(autoencoder.optimizer.lr)
        print ("Learning rate: " + str(lr))

        for index in range(NB_ITERATIONS):
            # Train Autoencoder
            if RAM_DECIMATE:
                X, y = load_X_y_RAM(videos_list, index, frames, [])
            else:
                X, y = load_X_y(videos_list, index, DATA_DIR, [])

            X_train = X[:, 0 : int(VIDEO_LENGTH/2)]
            y_train = X[:, int(VIDEO_LENGTH/2) :]
            loss.append(autoencoder.train_on_batch(X_train, y_train))

            arrow = int(index / (NB_ITERATIONS / 40))
            stdout.write("\rIter: " + str(index) + "/" + str(NB_ITERATIONS-1) + "  " +
                         "loss: " + str(loss[len(loss)-1]) +
                         "\t    [" + "{0}>".format("="*(arrow)))
            stdout.flush()

        if SAVE_GENERATED_IMAGES:
            # Save generated images to file
            predicted_images = autoencoder.predict(X_train, verbose=0)
            orig_image = arrange_images(X_train)
            truth_image = arrange_images(y_train)
            pred_image = arrange_images(predicted_images)
            orig_image = orig_image * 127.5 + 127.5
            pred_image = pred_image * 127.5 + 127.5
            truth_image = truth_image * 127.5 + 127.5
            if epoch == 0 :
                cv2.imwrite(os.path.join(GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_orig.png"), orig_image)
                cv2.imwrite(os.path.join(GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_truth.png"), truth_image)
            cv2.imwrite(os.path.join(GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_pred.png"), pred_image)

        # Run over validation data
        print ('')
        for index in range(NB_TEST_ITERATIONS):
            X, y = load_X_y(test_videos_list, index, TEST_DATA_DIR, [])
            X_train = X[:, 0 : int(VIDEO_LENGTH/2)]
            y_train = X[:, int(VIDEO_LENGTH/2) :]
            test_loss.append(autoencoder.test_on_batch(X_train, y_train))

            arrow = int(index / (NB_TEST_ITERATIONS / 40))
            stdout.write("\rIter: " + str(index) + "/" + str(NB_TEST_ITERATIONS-1) + "  " +
                         "test_loss: " + str(test_loss[len(test_loss)-1]) +
                         "\t    [" + "{0}>".format("="*(arrow)))
            stdout.flush()

        # then after each epoch/iteration
        avg_loss = sum(loss)/len(loss)
        avg_test_loss = sum(test_loss) / len(test_loss)
        logs = {'loss': avg_loss, 'test_loss' : avg_test_loss}
        TC.on_epoch_end(epoch, logs)

        # Log the losses
        with open(os.path.join(LOG_DIR, 'losses.json'), 'a') as log_file:
            log_file.write("{\"epoch\":%d, \"loss\":%f, \"test_loss\":%f};\n" % (epoch, avg_loss, avg_test_loss))

        print("\nAvg loss: " + str(avg_loss) + " Avg test loss: " + str(avg_test_loss))

        # Save model weights per epoch to file
        encoder.save_weights(os.path.join(CHECKPOINT_DIR, 'encoder_epoch_' + str(epoch) + '.h5'), True)
        decoder.save_weights(os.path.join(CHECKPOINT_DIR, 'decoder_epoch_' + str(epoch) + '.h5'), True)

        # Save predicted mask per epoch
        predicted_attn = mask_gen.predict(X_train, verbose=0)
        a_pred = np.reshape(predicted_attn, newshape=(BATCH_SIZE, 16, 14, 14, 1))
        np.save(os.path.join(ATTN_WEIGHTS_DIR, 'attention_weights_gen1_' + str(epoch) + '.npy'), a_pred)

    # Train Classifier
    if CLASSIFIER:
        print("Training Classifier...")
        for epoch in range(NB_EPOCHS_CLASS):
            print("\n\nEpoch ", epoch)
            c_loss = []
            test_c_loss = []

            # # Set learning rate every epoch
            LRS_clas.on_epoch_begin(epoch=epoch)
            lr = K.get_value(sclassifier.optimizer.lr)
            print("Learning rate: " + str(lr))
            print("c_loss_metrics: " + str(sclassifier.metrics_names))

            for index in range(NB_ITERATIONS):
                # Train Autoencoder
                if RAM_DECIMATE:
                    X, y = load_X_y_RAM(videos_list, index, frames, ped_action_classes)
                else:
                    X, y = load_X_y(videos_list, index, DATA_DIR, ped_action_classes)

                X_train = X[:, 0: int(VIDEO_LENGTH / 2)]
                y_true_class = y[:, CLASS_TARGET_INDEX]
                y_true_imgs = X[:, int(VIDEO_LENGTH / 2):]


                c_loss.append(sclassifier.train_on_batch(X_train, y_true_class))

                arrow = int(index / (NB_ITERATIONS / 30))
                stdout.write("\rIter: " + str(index) + "/" + str(NB_ITERATIONS - 1) + "  " +
                             "c_loss: " + str([ c_loss[len(c_loss) - 1][j]  for j in [0, 1, 2, 3, 4]]) + "  " +
                             "\t    [" + "{0}>".format("=" * (arrow)))
                stdout.flush()

            if SAVE_GENERATED_IMAGES:
                # Save generated images to file
                predicted_images = autoencoder.predict(X_train)
                ped_pred_class = sclassifier.predict(X_train, verbose=0)
                pred_seq = arrange_images(np.concatenate((X_train, predicted_images), axis=1))
                pred_seq = pred_seq * 127.5 + 127.5

                truth_image = arrange_images(y_true_imgs)
                truth_image = truth_image * 127.5 + 127.5
                font = cv2.FONT_HERSHEY_SIMPLEX
                y_orig_classes = y[:, 0: int(VIDEO_LENGTH / 2)]
                y_true_classes = y[:, int(VIDEO_LENGTH / 2):]
                # Add labels as text to the image

                for k in range(BATCH_SIZE):
                    for j in range(int(VIDEO_LENGTH / 2)):
                        class_num_past = np.argmax(y_orig_classes[k, j])
                        class_num_futr = np.argmax(y_true_classes[k, j])
                        class_num_y = np.argmax(ped_pred_class[k])
                        cv2.putText(pred_seq, simple_ped_set[class_num_past],
                                    (2 + j * (128), 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        cv2.putText(pred_seq, simple_ped_set[class_num_y],
                                    (2 + (j + 16) * (128), 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        cv2.putText(pred_seq, 'truth: ' + simple_ped_set[class_num_futr],
                                    (2 + (j + 16) * (128), 94 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        cv2.putText(truth_image, simple_ped_set[class_num_futr],
                                    (2 + j * (128), 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)

                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_cla_test_pred.png"), pred_seq)
                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_cla_test_truth.png"), truth_image)

            # Run over test data
            print('')
            for index in range(NB_TEST_ITERATIONS):
                X, y = load_X_y(test_videos_list, index, TEST_DATA_DIR, test_ped_action_classes)
                X_test = X[:, 0: int(VIDEO_LENGTH / 2)]
                y_true_class = y[:, CLASS_TARGET_INDEX]
                y_true_imgs = X[:, int(VIDEO_LENGTH / 2):]

                test_c_loss.append(sclassifier.test_on_batch(X_test, y_true_class))

                arrow = int(index / (NB_TEST_ITERATIONS / 40))
                stdout.write("\rIter: " + str(index) + "/" + str(NB_TEST_ITERATIONS - 1) + "  " +
                             "test_c_loss: " +  str([ test_c_loss[len(test_c_loss) - 1][j]  for j in [0, 1, 2, 3, 4]]))
                stdout.flush()

            # Save generated images to file
            test_predicted_images = autoencoder.predict(X_test)
            test_ped_pred_class = sclassifier.predict(X_test, verbose=0)
            orig_image = arrange_images(X_test)
            truth_image = arrange_images(y_true_imgs)
            pred_image = arrange_images(test_predicted_images)
            pred_image = pred_image * 127.5 + 127.5
            orig_image = orig_image * 127.5 + 127.5
            truth_image = truth_image * 127.5 + 127.5
            font = cv2.FONT_HERSHEY_SIMPLEX
            if epoch == 0:
                y_orig_classes = y[:, 0: int(VIDEO_LENGTH / 2)]
                y_true_classes = y[:, int(VIDEO_LENGTH / 2):]
                # Add labels as text to the image
                for k in range(BATCH_SIZE):
                    for j in range(int(VIDEO_LENGTH / 2)):
                        class_num_past = np.argmax(y_orig_classes[k, j])
                        class_num_futr = np.argmax(y_true_classes[k, j])
                        cv2.putText(orig_image, simple_ped_set[class_num_past],
                                    (2 + j * (128), 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        cv2.putText(truth_image, simple_ped_set[class_num_futr],
                                    (2 + j * (128), 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) +
                                         "_cla_test_orig.png"), orig_image)
                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) +
                                         "_cla_test_truth.png"), truth_image)

            # Add labels as text to the image
            for k in range(BATCH_SIZE):
                class_num_y = np.argmax(test_ped_pred_class[k])
                for j in range(int(VIDEO_LENGTH / 2)):
                    cv2.putText(pred_image, simple_ped_set[class_num_y],
                                (2 + j * (128), 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                                cv2.LINE_AA)
            cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_cla_test_pred.png"),
                        pred_image)

            predicted_attn = mask_gen.predict(X_train, verbose=0)
            a_pred = np.reshape(predicted_attn, newshape=(BATCH_SIZE, 16, 16, 16, 1))
            np.save(os.path.join(ATTN_WEIGHTS_DIR, 'attention_weights_cla_' + str(epoch) + '.npy'), a_pred)

            # then after each epoch/iteration
            avg_c_loss = np.mean(np.asarray(c_loss, dtype=np.float32), axis=0)
            avg_test_c_loss = np.mean(np.asarray(test_c_loss, dtype=np.float32), axis=0)

            loss_values = np.asarray(avg_c_loss.tolist() + avg_test_c_loss.tolist(), dtype=np.float32)
            c_loss_keys = ['c_' + metric for metric in classifier.metrics_names]
            test_c_loss_keys = ['c_test_' + metric for metric in classifier.metrics_names]

            loss_keys = c_loss_keys + test_c_loss_keys
            logs = dict(zip(loss_keys, loss_values))

            TC_cla.on_epoch_end(epoch, logs)

            # Log the losses
            with open(os.path.join(LOG_DIR, 'losses_cla.json'), 'a') as log_file:
                log_file.write("{\"epoch\":%d, %s;\n" % (epoch, logs))

            print("\nAvg c_loss: " + str(avg_c_loss) +
                  " Avg test_c_loss: " + str(avg_test_c_loss))

            # Save model weights per epoch to file
            # encoder.save_weights(os.path.join(CHECKPOINT_DIR, 'encoder_cla_epoch_' + str(epoch) + '.h5'), True)
            # decoder.save_weights(os.path.join(CHECKPOINT_DIR, 'decoder_cla_epoch_' + str(epoch) + '.h5'), True)
            classifier.save_weights(os.path.join(CHECKPOINT_DIR, 'classifier_cla_epoch_' + str(epoch) + '.h5'),
                                    True)

    # End TensorBoard Callback
    # TC.on_train_end('_')
    # TC_cla.on_train_end('_')

def test(ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS):

    # Setup test
    test_frames_source = hkl.load(os.path.join(TEST_DATA_DIR, 'sources_test_128.hkl'))
    test_videos_list = get_video_lists(frames_source=test_frames_source, stride=8)
    # Load test action annotations
    test_action_labels = hkl.load(os.path.join(TEST_DATA_DIR, 'annotations_test_128.hkl'))
    test_ped_action_classes, test_ped_class_count = get_action_classes(test_action_labels)
    print("Test Stats: " + str(test_ped_class_count))

    if not os.path.exists(TEST_RESULTS_DIR + '/truth/'):
        os.mkdir(TEST_RESULTS_DIR + '/truth/')
    if not os.path.exists(TEST_RESULTS_DIR + '/pred/'):
        os.mkdir(TEST_RESULTS_DIR + '/pred/')

    # Build the Spatio-temporal Autoencoder
    print("Creating models.")
    encoder = encoder_model()
    decoder = decoder_model()
    autoencoder = autoencoder_model(encoder, decoder)
    autoencoder.compile(loss="mean_absolute_error", optimizer=OPTIM_A)

    # Build stacked classifier
    if CLASSIFIER:
        classifier = pretrained_c3d()
        classifier.compile(loss="binary_crossentropy",
                           optimizer=OPTIM_C,
                           metrics=[metric_precision, metric_recall, metric_mpca, 'accuracy'])
        sclassifier = stacked_classifier_model(encoder, decoder, classifier)
        sclassifier.compile(loss="binary_crossentropy",
                            optimizer=OPTIM_C,
                            metrics=[metric_precision, metric_recall, metric_mpca, 'accuracy'])
        print(sclassifier.summary())

    if CLASSIFIER:
        run_utilities(encoder, decoder, autoencoder, classifier, ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS)
    else:
        run_utilities(encoder, decoder, autoencoder, 'classifier', ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS)

    n_test_videos = test_videos_list.shape[0]
    NB_TEST_ITERATIONS = int(n_test_videos / BATCH_SIZE)
    # NB_TEST_ITERATIONS = 1

    test_c_loss = []
    for index in range(NB_TEST_ITERATIONS):
        X, y = load_X_y(test_videos_list, index, TEST_DATA_DIR, test_ped_action_classes)
        X_test = X[:, 0: int(VIDEO_LENGTH / 2)]
        y_true_class = y[:, CLASS_TARGET_INDEX]
        y_true_imgs = X[:, int(VIDEO_LENGTH / 2):]

        test_c_loss.append(sclassifier.test_on_batch(X_test, y_true_class))

        arrow = int(index / (NB_TEST_ITERATIONS / 40))
        stdout.write("\rIter: " + str(index) + "/" + str(NB_TEST_ITERATIONS - 1) + "  " +
                     "test_c_loss: " + str([test_c_loss[len(test_c_loss) - 1][j] for j in [0, 1, 2, 3, 4]]))
        stdout.flush()

        # Save generated images to file
        test_predicted_images = autoencoder.predict(X_test)
        test_ped_pred_class = sclassifier.predict(X_test, verbose=0)
        pred_seq = arrange_images(np.concatenate((X_test, test_predicted_images), axis=1))
        pred_seq = pred_seq * 127.5 + 127.5

        truth_image = arrange_images(y_true_imgs)
        truth_image = truth_image * 127.5 + 127.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_orig_classes = y[:, 0: int(VIDEO_LENGTH / 2)]
        y_true_classes = y[:, int(VIDEO_LENGTH / 2):]
        # Add labels as text to the image

        for k in range(BATCH_SIZE):
            for j in range(int(VIDEO_LENGTH / 2)):
                class_num_past = np.argmax(y_orig_classes[k, j])
                class_num_futr = np.argmax(y_true_classes[k, j])
                class_num_y = np.argmax(test_ped_pred_class[k])
                cv2.putText(pred_seq, simple_ped_set[class_num_past],
                            (2 + j * (128), 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
                cv2.putText(pred_seq, simple_ped_set[class_num_y],
                            (2 + (j + 16) * (128), 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
                cv2.putText(pred_seq, 'truth: ' + simple_ped_set[class_num_futr],
                            (2 + (j + 16) * (128), 94 + k * 128), font, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
                cv2.putText(truth_image, simple_ped_set[class_num_futr],
                            (2 + j * (128), 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)

        cv2.imwrite(os.path.join(TEST_RESULTS_DIR + '/pred/', str(index) + "_cla_test_pred.png"), pred_seq)
        cv2.imwrite(os.path.join(TEST_RESULTS_DIR + '/truth/', str(index) + "_cla_test_truth.png"), truth_image)

        # then after each epoch/iteration
    avg_test_c_loss = np.mean(np.asarray(test_c_loss, dtype=np.float32), axis=0)

    print("\nAvg test_c_loss: " + str(avg_test_c_loss))
    print("\n Std: " + str(np.std(np.asarray(test_c_loss))))
    print("\n Variance: " + str(np.var(np.asarray(test_c_loss))))
    print("\n Mean: " + str(np.mean(np.asarray(test_c_loss))))
    print("\n Max: " + str(np.max(np.asarray(test_c_loss))))
    print("\n Min: " + str(np.min(np.asarray(test_c_loss))))
    np.save(os.path.join(TEST_RESULTS_DIR, 'L1_loss.npy'), test_c_loss)


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

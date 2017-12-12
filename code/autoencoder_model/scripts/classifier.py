from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hickle as hkl
import numpy as np

np.random.seed(2 ** 10)
from keras import backend as K
K.set_image_dim_ordering('tf')
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
    inputs = Input(shape=(10, 16, 16, 64))

    # 10x16x16
    convlstm_1 = ConvLSTM2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.5)(inputs)
    x = TimeDistributed(BatchNormalization())(convlstm_1)
    x = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_1 = TimeDistributed(Dropout(0.5))(x)

    flat_1 = TimeDistributed(Flatten())(out_1)
    aclstm_1 = GRU(units=16 * 16,
                   activation='tanh',
                   recurrent_dropout=0.5,
                   return_sequences=True)(flat_1)
    x = TimeDistributed(BatchNormalization())(aclstm_1)
    dense_1 = TimeDistributed(Dense(units=16 * 16, activation='softmax'))(x)
    a1_reshape = Reshape(target_shape=(10, 16, 16, 1))(dense_1)
    a1 = AttnLossLayer()(a1_reshape)
    dot_1 = multiply([out_1, a1])

    convlstm_2 = ConvLSTM2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.5)(dot_1)
    x = TimeDistributed(BatchNormalization())(convlstm_2)
    h_2 = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_2 = UpSampling3D(size=(1, 2, 2))(h_2)

    # 10x32x32
    convlstm_3 = ConvLSTM2D(filters=128,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.5)(out_2)
    x = TimeDistributed(BatchNormalization())(convlstm_3)
    h_3 = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_3 = UpSampling3D(size=(1, 2, 2))(h_3)

    # 10x64x64
    convlstm_4 = ConvLSTM2D(filters=32,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.5)(out_3)
    x = TimeDistributed(BatchNormalization())(convlstm_4)
    h_4 = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_4 = UpSampling3D(size=(1, 2, 2))(h_4)

    # 10x128x128
    convlstm_5 = ConvLSTM2D(filters=3,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.5)(out_4)
    predictions = TimeDistributed(Activation('tanh'))(convlstm_5)

    model = Model(inputs=inputs, outputs=predictions)

    return model


def classifier_model():
    inputs = Input(shape=(10, 128, 128, 3))
    conv_1 = ConvLSTM2D(filters=32,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding="same",
                        return_sequences=True,
                        recurrent_dropout=0.5)(inputs)
    conv_1 = TimeDistributed(LeakyReLU(alpha=0.2))(conv_1)
    conv_1 = TimeDistributed(Dropout(0.5))(conv_1)

    conv_2 = ConvLSTM2D(filters=64,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding="same",
                        return_sequences=True,
                        recurrent_dropout=0.5)(conv_1)
    conv_2 = TimeDistributed(BatchNormalization())(conv_2)
    conv_2 = TimeDistributed(LeakyReLU(alpha=0.2))(conv_2)
    conv_2 = TimeDistributed(Dropout(0.5))(conv_2)

    conv_3 = ConvLSTM2D(filters=128,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding="same",
                        return_sequences=True,
                        recurrent_dropout=0.5)(conv_2)
    conv_3 = TimeDistributed(BatchNormalization())(conv_3)
    conv_3 = TimeDistributed(LeakyReLU(alpha=0.2))(conv_3)
    conv_3 = TimeDistributed(Dropout(0.5))(conv_3)

    conv_4 = ConvLSTM2D(filters=256,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding="same",
                        return_sequences=False,
                        recurrent_dropout=0.5)(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = LeakyReLU(alpha=0.2)(conv_4)
    conv_4 = Dropout(0.5)(conv_4)

    flat_1 = Flatten()(conv_4)
    dense_1 = Dense(units=512, activation='tanh')(flat_1)
    dense_1 = Dropout(0.5)(dense_1)
    dense_2 = Dense(units=len(driver_actions), activation='sigmoid')(dense_1)
    dense_3 = TimeDistributed(Dense(units=len(ped_actions), activation='sigmoid'))(dense_1)

    model = Model(inputs=inputs, outputs=[dense_2, dense_3])

    return model


def conv_classifier_model():
    inputs = Input(shape=(10, 128, 128, 3))
    conv_1 = TimeDistributed(Conv2D(filters=32,
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding="same",
                                    kernel_regularizer=regularizers.l2()))(inputs)
    conv_1 = TimeDistributed(LeakyReLU(alpha=0.2))(conv_1)
    conv_1 = TimeDistributed(Dropout(0.5))(conv_1)

    conv_2 = TimeDistributed(Conv2D(filters=64,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding="same",
                                    kernel_regularizer=regularizers.l2()))(conv_1)
    conv_2 = TimeDistributed(BatchNormalization())(conv_2)
    conv_2 = TimeDistributed(LeakyReLU(alpha=0.2))(conv_2)
    conv_2 = TimeDistributed(Dropout(0.5))(conv_2)

    conv_3 = TimeDistributed(Conv2D(filters=128,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding="same",
                                    kernel_regularizer=regularizers.l2()))(conv_2)
    conv_3 = TimeDistributed(BatchNormalization())(conv_3)
    conv_3 = TimeDistributed(LeakyReLU(alpha=0.2))(conv_3)
    conv_3 = TimeDistributed(Dropout(0.5))(conv_3)

    conv_4 = TimeDistributed(Conv2D(filters=128,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding="same",
                                    kernel_regularizer=regularizers.l2()))(conv_3)
    conv_4 = TimeDistributed(BatchNormalization())(conv_4)
    conv_4 = TimeDistributed(LeakyReLU(alpha=0.2))(conv_4)
    conv_4 = TimeDistributed(Dropout(0.5))(conv_4)

    flat_1 = TimeDistributed(Flatten())(conv_4)
    dense_1 = TimeDistributed(Dense(units=512, activation='tanh'))(flat_1)
    dense_1 = TimeDistributed(Dropout(0.5))(dense_1)
    dense_2 = TimeDistributed(Dense(units=len(joint_action_set), activation='softmax'))(dense_1)

    model = Model(inputs=inputs, outputs=dense_2)

    return model


def lstm_classifier():
    inputs = Input(shape=(10, 128, 128, 3))

    conv_1 = TimeDistributed(Conv2D(filters=32,
                                    kernel_size=(5, 5),
                                    strides=(2, 2),
                                    padding="same"))(inputs)
    conv_1 = TimeDistributed(LeakyReLU(alpha=0.2))(conv_1)
    conv_1 = TimeDistributed(Dropout(0.5))(conv_1)

    conv_2 = TimeDistributed(Conv2D(filters=64,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding="same"))(conv_1)
    conv_2 = TimeDistributed(BatchNormalization())(conv_2)
    conv_2 = TimeDistributed(LeakyReLU(alpha=0.2))(conv_2)
    conv_2 = TimeDistributed(Dropout(0.5))(conv_2)

    conv_3 = TimeDistributed(Conv2D(filters=128,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding="same"))(conv_2)
    conv_3 = TimeDistributed(BatchNormalization())(conv_3)
    conv_3 = TimeDistributed(LeakyReLU(alpha=0.2))(conv_3)
    conv_3 = TimeDistributed(Dropout(0.5))(conv_3)

    flat_1 = TimeDistributed(Flatten())(conv_3)
    lstm_1 = LSTM(units=256,
                  return_sequences=True,
                  recurrent_dropout=0.5)(flat_1)
    lstm_1 = TimeDistributed(BatchNormalization())(lstm_1)
    lstm_1 = TimeDistributed(LeakyReLU(alpha=0.2))(lstm_1)

    lstm_2 = LSTM(units=256,
                  return_sequences=True,
                  recurrent_dropout=0.5)(lstm_1)
    lstm_2 = TimeDistributed(BatchNormalization())(lstm_2)
    lstm_2 = TimeDistributed(LeakyReLU(alpha=0.2))(lstm_2)

    dense_1 = TimeDistributed(Dense(units=len(driver_actions), activation='sigmoid'))(lstm_2)
    dense_2 = TimeDistributed(Dense(units=len(ped_actions), activation='sigmoid'))(lstm_2)

    model = Model(inputs=inputs, outputs=[dense_1, dense_2])

    return model

def shuffled_conv_classifier_model():
    inputs = Input(shape=(128, 128, 3))
    conv_1 = Conv2D(filters=32,
                    kernel_size=(5, 5),
                    strides=(2, 2),
                    padding="same",
                    kernel_regularizer=regularizers.l2())(inputs)
    conv_1 = LeakyReLU(alpha=0.2)(conv_1)
    conv_1 = Dropout(0.5)(conv_1)

    conv_2 = Conv2D(filters=64,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="same",
                    kernel_regularizer=regularizers.l2())(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = LeakyReLU(alpha=0.2)(conv_2)
    conv_2 = Dropout(0.5)(conv_2)

    conv_3 = Conv2D(filters=128,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="same",
                    kernel_regularizer=regularizers.l2())(conv_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = LeakyReLU(alpha=0.2)(conv_3)
    conv_3 = Dropout(0.5)(conv_3)

    conv_4 = Conv2D(filters=128,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="same",
                    kernel_regularizer=regularizers.l2())(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = LeakyReLU(alpha=0.2)(conv_4)
    conv_4 = Dropout(0.5)(conv_4)

    flat_1 = Flatten()(conv_4)
    dense_1 = Dense(units=512, activation='tanh')(flat_1)
    dense_1 = Dropout(0.5)(dense_1)
    dense_2 = Dense(units=5, activation='softmax')(dense_1)

    model = Model(inputs=inputs, outputs=dense_2)

    return model


def set_trainability(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def autoencoder_model(encoder, decoder):
    model = Sequential()
    model.add(encoder)
    model.add(decoder)
    return model


def action_model(encoder, decoder, classifier):
    inputs = Input(shape=(int(VIDEO_LENGTH/2), 128, 128, 3))
    set_trainability(encoder, False)
    z = encoder(inputs)
    set_trainability(decoder, False)
    images = decoder(z)
    predictions = classifier(images)

    # model = Model(inputs=inputs, outputs=[images, predictions])
    model = Model(inputs=inputs, outputs=predictions)

    return model


def stacked_classifier_model(encoder, decoder, classifier):
    input = Input(shape=(VIDEO_LENGTH-10, 128, 128, 3))
    set_trainability(encoder, False)
    z = encoder(input)
    set_trainability(decoder, False)
    future = decoder(z)
    actions = classifier(future)

    model = Model(inputs=input, outputs=actions)

    return model


def combine_images(X, y, generated_images):
    # Unroll all generated video frames
    n_frames = generated_images.shape[0] * generated_images.shape[1]
    frames = np.zeros((n_frames,) + generated_images.shape[2:], dtype=generated_images.dtype)

    frame_index = 0
    for i in range(generated_images.shape[0]):
        for j in range(generated_images.shape[1]):
            frames[frame_index] = generated_images[i, j]
            frame_index += 1

    num = frames.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = frames.shape[1:]
    image = np.zeros((height * shape[0], width * shape[1], shape[2]), dtype=generated_images.dtype)
    for index, img in enumerate(frames):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = img

    n_frames = X.shape[0] * X.shape[1]
    orig_frames = np.zeros((n_frames,) + X.shape[2:], dtype=X.dtype)

    # Original frames
    frame_index = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            orig_frames[frame_index] = X[i, j]
            frame_index += 1

    num = orig_frames.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = orig_frames.shape[1:]
    orig_image = np.zeros((height * shape[0], width * shape[1], shape[2]), dtype=X.dtype)
    for index, img in enumerate(orig_frames):
        i = int(index / width)
        j = index % width
        orig_image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = img

    # Ground truth
    truth_frames = np.zeros((n_frames,) + y.shape[2:], dtype=y.dtype)
    frame_index = 0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            truth_frames[frame_index] = y[i, j]
            frame_index += 1

    num = truth_frames.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = truth_frames.shape[1:]
    truth_image = np.zeros((height * shape[0], width * shape[1], shape[2]), dtype=y.dtype)
    for index, img in enumerate(truth_frames):
        i = int(index / width)
        j = index % width
        truth_image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = img

    return orig_image, truth_image, image


def load_weights(weights_file, model):
    model.load_weights(weights_file)


def run_utilities(encoder, decoder, autoencoder, classifier, ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS):
# def run_utilities(encoder, decoder, autoencoder, ENC_WEIGHTS, DEC_WEIGHTS):
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

        # exit(0)

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


def load_X_y(videos_list, index, data_dir, driver_action_cats, ped_action_cats):
    X = np.zeros((BATCH_SIZE, VIDEO_LENGTH,) + IMG_SIZE)
    y1 = []
    y2 = []
    for i in range(BATCH_SIZE):
        y1_per_vid = []
        y2_per_vid = []
        for j in range(VIDEO_LENGTH):
            frame_number = (videos_list[(index*BATCH_SIZE + i), j])
            filename = "frame_" + str(frame_number) + ".png"
            im_file = os.path.join(data_dir, filename)
            try:
                frame = cv2.imread(im_file, cv2.IMREAD_COLOR)
                X[i, j] = (frame.astype(np.float32) - 127.5) / 127.5
            except AttributeError as e:
                print (im_file)
                print (e)
            if (len(driver_action_cats) != 0):
                try:
                    y1_per_vid.append(driver_action_cats[frame_number - 1])
                except IndexError as e:
                    print (frame_number)
                    print (e)
            if (len(ped_action_cats) != 0):
                try:
                    y2_per_vid.append(ped_action_cats[frame_number - 1])
                except IndexError as e:
                    print(frame_number)
                    print(e)
        if (len(driver_action_cats) != 0):
            y1.append(y1_per_vid)
        if (len(ped_action_cats) != 0):
            y2.append(y2_per_vid)
    return X, np.asarray(y1), np.asarray(y2)


def get_action_classes(action_labels):
    # Load labesl into categorical 1-hot vectors
    print("Loading annotations.")
    driver_action_class = []
    ped_action_class = []
    # action_class = []
    for i in range(len(action_labels)):
        action_dict = dict(ele.split(':') for ele in action_labels[i].split(', ')[2:])
        if ',' in action_dict['Driver']:
            driver_action_nums = driver_actions.index(action_dict['Driver'].split(',')[0])
            encoded_driver_action = to_categorical(driver_action_nums, len(driver_actions))
            driver_action_class.append(encoded_driver_action.T)
        else:
            driver_action_nums = driver_actions.index(action_dict['Driver'])
            encoded_driver_action = to_categorical(driver_action_nums, len(driver_actions))
            driver_action_class.append(encoded_driver_action.T)
        a = action_dict.values()[0:(len(action_dict.values()) - 1)]
        a_clean = []
        for j in range(len(a)):
            if ',' in a[j]:
                splits = a[j].split(',')
                for k in range(len(splits)):
                    a_clean.append(splits[k])
            else:
                a_clean.append(a[j])
        ped_action_per_frame = list(set(a_clean))
        encoded_ped_action = np.zeros(shape=(1, len(ped_actions)), dtype=np.float32)
        # print (ped_action_per_frame)
        for action in ped_action_per_frame:
            # Offset ped action from driver action
            ped_action = ped_actions.index(action)
            # Add all unique categorical one-hot vectors
            encoded_ped_action = encoded_ped_action + to_categorical(ped_action, len(ped_actions))

        ped_action_class.append(encoded_ped_action.T)
        # action_class.append((encoded_driver_action + encoded_ped_action).T)

    driver_action_class = np.asarray(driver_action_class)
    driver_action_class = np.reshape(driver_action_class, newshape=(driver_action_class.shape[0:2]))
    ped_action_class = np.asarray(ped_action_class)
    ped_action_class = np.reshape(ped_action_class, newshape=(ped_action_class.shape[0:2]))
    # action_class = np.asarray(action_class)
    # action_class = np.reshape(action_class, newshape=(action_class.shape[0:2]))

    # print (driver_action_class.shape)
    # print (ped_action_class.shape)
    # print (action_class.shape)
    # print (np.where(driver_action_class > 1))
    # print (np.where(ped_action_class > 1))
    # print (np.where(action_class>1))
    # print (driver_action_class[0])
    # print (ped_action_class[0])
    # print (action_class[0])
    # exit(0)

    return driver_action_class, ped_action_class


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

    if SHUFFLE:
        # Shuffle images to aid generalization
        videos_list = np.random.permutation(videos_list)

    # Load actions from annotations
    action_labels = hkl.load(os.path.join(DATA_DIR, 'annotations_train_128.hkl'))
    driver_action_classes, ped_action_classes = get_action_classes(action_labels=action_labels)

    # Setup validation
    val_frames_source = hkl.load(os.path.join(VAL_DATA_DIR, 'sources_val_128.hkl'))
    val_videos_list = get_video_lists(frames_source=val_frames_source, stride=VIDEO_LENGTH)
    # Load val action annotations
    val_action_labels = hkl.load(os.path.join(VAL_DATA_DIR, 'annotations_val_128.hkl'))
    val_driver_action_classes, val_ped_action_classes = get_action_classes(val_action_labels)

    # Build the Spatio-temporal Autoencoder
    print ("Creating models.")
    encoder = encoder_model()
    decoder = decoder_model()

    # Build attention layer output
    intermediate_decoder = Model(inputs=decoder.layers[0].input, outputs=decoder.layers[10].output)
    mask_gen_1 = Sequential()
    mask_gen_1.add(encoder)
    mask_gen_1.add(intermediate_decoder)
    mask_gen_1.compile(loss='mean_squared_error', optimizer=OPTIM_G)

    autoencoder = autoencoder_model(encoder, decoder)
    autoencoder.compile(loss="mean_squared_error", optimizer=OPTIM_A)

    # Build stacked classifier
    classifier = classifier_model()
    classifier.compile(loss=["categorical_crossentropy", "categorical_crossentropy"],
                             optimizer=OPTIM_C, metrics=['accuracy'])
    sclassifier = stacked_classifier_model(encoder, decoder, classifier)
    sclassifier.compile(loss=["categorical_crossentropy", "categorical_crossentropy"],
                              optimizer=OPTIM_C,
                              metrics=['accuracy'])

    run_utilities(encoder, decoder, autoencoder, classifier, ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS)

    print (sclassifier.summary())

    n_videos = videos_list.shape[0]
    n_val_videos = val_videos_list.shape[0]
    NB_ITERATIONS = int(n_videos/BATCH_SIZE)
    # NB_ITERATIONS = 1
    NB_VAL_ITERATIONS = int(n_val_videos/BATCH_SIZE)

    # Setup TensorBoard Callback
    TC = tb_callback.TensorBoard(log_dir=TF_LOG_DIR, histogram_freq=0, write_graph=False, write_images=False)
    TC_cla = tb_callback.TensorBoard(log_dir=TF_LOG_CLA_DIR, histogram_freq=0, write_graph=False, write_images=False)
    LRS = lrs_callback.LearningRateScheduler(schedule=schedule)
    LRS.set_model(autoencoder)


    print ("Beginning Training.")
    # Begin Training
    for epoch in range(NB_EPOCHS_AUTOENCODER):
        print("\n\nEpoch ", epoch)
        loss = []
        val_loss = []

        # Set learning rate every epoch
        LRS.on_epoch_begin(epoch=epoch)
        lr = K.get_value(autoencoder.optimizer.lr)
        print ("Learning rate: " + str(lr))

        for index in range(NB_ITERATIONS):
            # Train Autoencoder
            X, y1, y2 = load_X_y(videos_list, index, DATA_DIR, [], [])
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
            orig_image, truth_image, pred_image = combine_images(X_train, y_train, predicted_images)
            pred_image = pred_image * 127.5 + 127.5
            orig_image = orig_image * 127.5 + 127.5
            truth_image = truth_image * 127.5 + 127.5
            if epoch == 0 :
                cv2.imwrite(os.path.join(GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_orig.png"), orig_image)
                cv2.imwrite(os.path.join(GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_truth.png"), truth_image)
            cv2.imwrite(os.path.join(GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_pred.png"), pred_image)

        # Run over validation data
        for index in range(NB_VAL_ITERATIONS):
            X, y1, y2 = load_X_y(val_videos_list, index, VAL_DATA_DIR, [], [])
            X_train = X[:, 0 : int(VIDEO_LENGTH/2)]
            y_train = X[:, int(VIDEO_LENGTH/2) :]
            val_loss.append(autoencoder.test_on_batch(X_train, y_train))

            arrow = int(index / (NB_VAL_ITERATIONS / 40))
            stdout.write("\rIter: " + str(index) + "/" + str(NB_VAL_ITERATIONS-1) + "  " +
                         "val_loss: " + str(val_loss[len(val_loss)-1]) +
                         "\t    [" + "{0}>".format("="*(arrow)))
            stdout.flush()

        # then after each epoch/iteration
        avg_loss = sum(loss)/len(loss)
        avg_val_loss = sum(val_loss) / len(val_loss)
        logs = {'loss': avg_loss, 'val_loss' : avg_val_loss}
        TC.on_epoch_end(epoch, logs)

        # Log the losses
        with open(os.path.join(LOG_DIR, 'losses.json'), 'a') as log_file:
            log_file.write("{\"epoch\":%d, \"d_loss\":%f};\n" % (epoch, avg_loss))

        print("\nAvg loss: " + str(avg_loss) + " Avg val loss: " + str(avg_val_loss))

        # Save model weights per epoch to file
        encoder.save_weights(os.path.join(CHECKPOINT_DIR, 'encoder_epoch_' + str(epoch) + '.h5'), True)
        decoder.save_weights(os.path.join(CHECKPOINT_DIR, 'decoder_epoch_' + str(epoch) + '.h5'), True)

        # Save predicted mask per epoch
        predicted_attn = mask_gen_1.predict(X_train, verbose=0)
        a_pred = np.reshape(predicted_attn, newshape=(10, 10, 16, 16, 1))
        np.save(os.path.join(ATTN_WEIGHTS_DIR, 'attention_weights_gen1_' + str(epoch) + '.npy'), a_pred)

    # Train AAE
    if CLASSIFIER:
        print("Training Classifier...")
        for epoch in range(NB_EPOCHS_CLASS):
            print("\n\nEpoch ", epoch)
            c_loss = []
            val_c_loss = []

            # # Set learning rate every epoch
            # LRS.on_epoch_begin(epoch=epoch)
            lr = K.get_value(autoencoder.optimizer.lr)
            print("Learning rate: " + str(lr))
            print("c_loss_metrics: " + str(classifier.metrics_names))

            for index in range(NB_ITERATIONS):
                # Train Autoencoder
                X, y1, y2 = load_X_y(videos_list, index, DATA_DIR, driver_action_classes, ped_action_classes)
                X_train = X[:, 0: int(VIDEO_LENGTH / 2)]
                y_true_imgs = X[:, int(VIDEO_LENGTH / 2):]
                y1_true_classes = y1[:, int(VIDEO_LENGTH / 2):]
                y2_true_classes = y2[:, int(VIDEO_LENGTH / 2):]

                c_loss.append(sclassifier.train_on_batch(X_train, [y1_true_classes, y2_true_classes]))

                arrow = int(index / (NB_ITERATIONS / 30))
                stdout.write("\rIter: " + str(index) + "/" + str(NB_ITERATIONS - 1) + "  " +
                             "c_loss: " + str([ c_loss[len(c_loss) - 1][j]  for j in [0, 3, 4]]) + "  " +
                             "\t    [" + "{0}>".format("=" * (arrow)))
                stdout.flush()

            if SAVE_GENERATED_IMAGES:
                # Save generated images to file
                predicted_images = autoencoder.predict(X_train)
                driver_pred_classes, ped_pred_classes = sclassifier.predict(X_train, verbose=0)
                orig_image, truth_image, pred_image = combine_images(X_train, y_true_imgs, predicted_images)
                pred_image = pred_image * 127.5 + 127.5
                orig_image = orig_image * 127.5 + 127.5
                truth_image = truth_image * 127.5 + 127.5
                font = cv2.FONT_HERSHEY_SIMPLEX
                if epoch == 0:
                    y1_orig_classes = y1[:, 0: int(VIDEO_LENGTH / 2)]
                    y2_orig_classes = y2[:, 0: int(VIDEO_LENGTH / 2)]
                    # Add labels as text to the image
                    for k in range(BATCH_SIZE):
                        for j in range(int(VIDEO_LENGTH / 2)):
                                class_num_past_y1 = np.argmax(y1_orig_classes[k, j])
                                class_num_past_y2 = np.argmax(y2_orig_classes[k, j])
                                class_num_futr_y1 = np.argmax(y1_true_classes[k, j])
                                class_num_futr_y2 = np.argmax(y2_true_classes[k, j])
                                cv2.putText(orig_image, "Car: " + driver_actions[class_num_past_y1],
                                            (2 + j * (128), 120 + k * 128), font, 0.3, (255, 255, 255), 1,
                                            cv2.LINE_AA)
                                cv2.putText(orig_image, "Ped: " + ped_actions[class_num_past_y2],
                                            (2 + j * (128), 120 - 15 + k * 128), font, 0.3, (255, 255, 255), 1,
                                            cv2.LINE_AA)
                                cv2.putText(truth_image, "Car: " + driver_actions[class_num_futr_y1],
                                            (2 + j * (128), 120 + k * 128), font, 0.3, (255, 255, 255), 1,
                                            cv2.LINE_AA)
                                cv2.putText(truth_image, "Ped: " + ped_actions[class_num_futr_y2],
                                            (2 + j * (128), 120 - 15 + k * 128), font, 0.3, (255, 255, 255), 1,
                                            cv2.LINE_AA)
                    cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) +
                                             "_cla_orig.png"), orig_image)
                    cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) +
                                             "_cla_truth.png"), truth_image)

                # Add labels as text to the image
                for k in range(BATCH_SIZE):
                    for j in range(int(VIDEO_LENGTH / 2)):
                            class_num_y1 = np.argmax(driver_pred_classes[k, j])
                            class_num_y2 = np.argmax(ped_pred_classes[k, j])
                            cv2.putText(pred_image,  "Car: " + driver_actions[class_num_y1],
                                        (2 + j * (128), 120 + k * 128), font, 0.3, (255, 255, 255), 1,
                                        cv2.LINE_AA)
                            cv2.putText(pred_image, "Ped: " +  ped_actions[class_num_y2],
                                        (2 + j * (128), 120 - 15 + k * 128), font, 0.3, (255, 255, 255), 1,
                                        cv2.LINE_AA)
                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_cla_pred.png"),
                            pred_image)

            # Run over validation data
            print('')
            for index in range(NB_VAL_ITERATIONS):
                X, y1, y2 = load_X_y(val_videos_list, index, VAL_DATA_DIR, val_driver_action_classes, val_ped_action_classes)
                X_val = X[:, 0: int(VIDEO_LENGTH / 2)]
                y1_true_classes = y1[:, int(VIDEO_LENGTH / 2):]
                y2_true_classes = y2[:, int(VIDEO_LENGTH / 2):]
                y_true_imgs = X[:, int(VIDEO_LENGTH / 2):]

                val_c_loss.append(sclassifier.test_on_batch(X_val, [y1_true_classes, y2_true_classes]))

                arrow = int(index / (NB_VAL_ITERATIONS / 40))
                stdout.write("\rIter: " + str(index) + "/" + str(NB_VAL_ITERATIONS - 1) + "  " +
                             "val_c_loss: " +  str([ val_c_loss[len(val_c_loss) - 1][j]  for j in [0, 3, 4]]))
                stdout.flush()

            # Save generated images to file
            predicted_images = autoencoder.predict(X_val)
            driver_pred_classes, ped_pred_classes = sclassifier.predict(X_val, verbose=0)
            orig_image, truth_image, pred_image = combine_images(X_val, y_true_imgs, predicted_images)
            pred_image = pred_image * 127.5 + 127.5
            orig_image = orig_image * 127.5 + 127.5
            truth_image = truth_image * 127.5 + 127.5
            font = cv2.FONT_HERSHEY_SIMPLEX
            if epoch == 0:
                y1_orig_classes = y1[:, 0: int(VIDEO_LENGTH / 2)]
                y2_orig_classes = y2[:, 0: int(VIDEO_LENGTH / 2)]
                # Add labels as text to the image
                for k in range(BATCH_SIZE):
                    for j in range(int(VIDEO_LENGTH / 2)):
                        class_num_past_y1 = np.argmax(y1_orig_classes[k, j])
                        class_num_past_y2 = np.argmax(y2_orig_classes[k, j])
                        class_num_futr_y1 = np.argmax(y1_true_classes[k, j])
                        class_num_futr_y2 = np.argmax(y2_true_classes[k, j])
                        cv2.putText(orig_image, "Car: " + driver_actions[class_num_past_y1],
                                    (2 + j * (128), 120 + k * 128), font, 0.3, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        cv2.putText(orig_image, "Ped: " + ped_actions[class_num_past_y2],
                                    (2 + j * (128), 120 - 15 + k * 128), font, 0.3, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        cv2.putText(truth_image, "Car: " + driver_actions[class_num_futr_y1],
                                    (2 + j * (128), 120 + k * 128), font, 0.3, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        cv2.putText(truth_image, "Ped: " + ped_actions[class_num_futr_y2],
                                    (2 + j * (128), 120 - 15 + k * 128), font, 0.3, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) +
                                         "_cla_val_orig.png"), orig_image)
                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) +
                                         "_cla_val_truth.png"), truth_image)

            # Add labels as text to the image
            for k in range(BATCH_SIZE):
                for j in range(int(VIDEO_LENGTH / 2)):
                    class_num_y1 = np.argmax(driver_pred_classes[k, j])
                    class_num_y2 = np.argmax(ped_pred_classes[k, j])
                    cv2.putText(pred_image, "Car: " + driver_actions[class_num_y1],
                                (2 + j * (128), 120 + k * 128), font, 0.3, (255, 255, 255), 1,
                                cv2.LINE_AA)
                    cv2.putText(pred_image, "Ped: " + ped_actions[class_num_y2],
                                (2 + j * (128), 120 - 15 + k * 128), font, 0.3, (255, 255, 255), 1,
                                cv2.LINE_AA)
            cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_cla_val_pred.png"),
                        pred_image)

            predicted_attn = mask_gen_1.predict(X_train, verbose=0)
            a_pred = np.reshape(predicted_attn, newshape=(BATCH_SIZE, VIDEO_LENGTH-10, 16, 16, 1))
            np.save(os.path.join(ATTN_WEIGHTS_DIR, 'attention_weights_cla_gen1_' + str(epoch) + '.npy'), a_pred)

            # then after each epoch/iteration
            avg_c_loss = np.mean(np.asarray(c_loss, dtype=np.float32), axis=0)
            avg_val_c_loss = np.mean(np.asarray(val_c_loss, dtype=np.float32), axis=0)

            loss_values = np.asarray(avg_c_loss.tolist() + avg_val_c_loss.tolist(), dtype=np.float32)
            c_loss_keys = ['c_' + metric for metric in classifier.metrics_names]
            val_c_loss_keys = ['c_val_' + metric for metric in classifier.metrics_names]

            loss_keys = c_loss_keys + val_c_loss_keys
            logs = dict(zip(loss_keys, loss_values))

            TC_cla.on_epoch_end(epoch, logs)

            # Log the losses
            with open(os.path.join(LOG_DIR, 'losses_aae.json'), 'a') as log_file:
                log_file.write("{\"epoch\":%d, %s;\n" % (epoch, logs))

            print("\nAvg c_loss: " + str(avg_c_loss) +
                  " Avg val_c_loss: " + str(avg_val_c_loss))

            # Save model weights per epoch to file
            encoder.save_weights(os.path.join(CHECKPOINT_DIR, 'encoder_cla_epoch_' + str(epoch) + '.h5'), True)
            decoder.save_weights(os.path.join(CHECKPOINT_DIR, 'decoder_cla_epoch_' + str(epoch) + '.h5'), True)
            classifier.save_weights(os.path.join(CHECKPOINT_DIR, 'classifier_cla_epoch_' + str(epoch) + '.h5'),
                                    True)

    # End TensorBoard Callback
    # TC.on_train_end('_')
    # TC_cla.on_train_end('_')

def test(ENC_WEIGHTS, DEC_WEIGHTS):

    # Create models
    print ("Creating models.")
    encoder = encoder_model()
    decoder = decoder_model()
    autoencoder = autoencoder_model(encoder, decoder)

    run_utilities(encoder, decoder, autoencoder, ENC_WEIGHTS, DEC_WEIGHTS)
    autoencoder.compile(loss='mean_squared_error', optimizer=OPTIM_A)

    for i in range(len(decoder.layers)):
        print (decoder.layers[i], str(i))

    exit(0)

    def build_intermediate_model(encoder, decoder):
        # convlstm-13, conv3d-25
        intermediate_decoder_1 = Model(inputs=decoder.layers[0].input, outputs=decoder.layers[21].output)
        intermediate_decoder_2 = Model(inputs=decoder.layers[0].input, outputs=decoder.layers[27].output)

        imodel_1 = Sequential()
        imodel_1.add(encoder)
        imodel_1.add(intermediate_decoder_1)

        imodel_2 = Sequential()
        imodel_2.add(encoder)
        imodel_2.add(intermediate_decoder_2)

        return imodel_1, imodel_2

    imodel_1, imodel_2 = build_intermediate_model(encoder, decoder)
    imodel_1.compile(loss='mean_squared_error', optimizer=OPTIM)
    imodel_2.compile(loss='mean_squared_error', optimizer=OPTIM)

    # Build video progressions
    frames_source = hkl.load(os.path.join(TEST_DATA_DIR, 'sources_test_128.hkl'))
    videos_list = []
    start_frame_index = 1
    end_frame_index = VIDEO_LENGTH + 1
    while (end_frame_index <= len(frames_source)):
        frame_list = frames_source[start_frame_index:end_frame_index]
        if (len(set(frame_list)) == 1):
            videos_list.append(range(start_frame_index, end_frame_index))
            start_frame_index = start_frame_index + VIDEO_LENGTH
            end_frame_index = end_frame_index + VIDEO_LENGTH
        else:
            start_frame_index = end_frame_index - 1
            end_frame_index = start_frame_index + VIDEO_LENGTH

    videos_list = np.asarray(videos_list, dtype=np.int32)
    n_videos = videos_list.shape[0]

    # Test model by making predictions
    loss = []
    NB_ITERATIONS = int(n_videos / BATCH_SIZE)
    for index in range(NB_ITERATIONS):
        # Test Autoencoder
        X = load_X(videos_list, index, TEST_DATA_DIR)
        X_test = X[:, 0: int(VIDEO_LENGTH / 2)]
        y_test = X[:, int(VIDEO_LENGTH / 2):]
        loss.append(autoencoder.test_on_batch(X_test, y_test))
        y_pred = autoencoder.predict_on_batch(X_test)
        a_pred_1 = imodel_1.predict_on_batch(X_test)
        a_pred_2 = imodel_2.predict_on_batch(X_test)

        arrow = int(index / (NB_ITERATIONS / 40))
        stdout.write("\rIteration: " + str(index) + "/" + str(NB_ITERATIONS - 1) + "  " +
                     "loss: " + str(loss[len(loss) - 1]) +
                     "\t    [" + "{0}>".format("=" * (arrow)))
        stdout.flush()

        orig_image, truth_image, pred_image = combine_images(X_test, y_test, y_pred)
        pred_image = pred_image * 127.5 + 127.5
        orig_image = orig_image * 127.5 + 127.5
        truth_image = truth_image * 127.5 + 127.5

        cv2.imwrite(os.path.join(TEST_RESULTS_DIR, str(index) + "_orig.png"), orig_image)
        cv2.imwrite(os.path.join(TEST_RESULTS_DIR, str(index) + "_truth.png"), truth_image)
        cv2.imwrite(os.path.join(TEST_RESULTS_DIR, str(index) + "_pred.png"), pred_image)

        #------------------------------------------
        a_pred_1 = np.reshape(a_pred_1, newshape=(10, 10, 64, 64, 1))
        np.save(os.path.join(TEST_RESULTS_DIR, 'attention_weights_1_' + str(index) +'.npy'), a_pred_1)
        np.save(os.path.join(TEST_RESULTS_DIR, 'attention_weights_2_' + str(index) + '.npy'), a_pred_2)
        # orig_image, truth_image, pred_image = combine_images(X_test, y_test, a_pred_1)
        # pred_image = (pred_image*100) * 127.5 + 127.5
        # y_pred = y_pred * 127.5 + 127.5
        # np.save(os.path.join(TEST_RESULTS_DIR, 'attention_weights_' + str(index) + '.npy'), y_pred)
        # cv2.imwrite(os.path.join(TEST_RESULTS_DIR, str(index) + "_attn_1.png"), pred_image)

        # a_pred_2 = np.reshape(a_pred_2, newshape=(10, 10, 16, 16, 1))
        # with open('attention_weights.txt', mode='w') as file:
        #     file.write(str(a_pred_2[0, 4]))
        # orig_image, truth_image, pred_image = combine_images(X_test, y_test, a_pred_2)
        # pred_image = (pred_image*100) * 127.5 + 127.5
        # cv2.imwrite(os.path.join(TEST_RESULTS_DIR, str(index) + "_attn_2.png"), pred_image)

    avg_loss = sum(loss) / len(loss)
    print("\nAvg loss: " + str(avg_loss))


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
             DEC_WEIGHTS=args.dec_weights)

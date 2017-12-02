from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hickle as hkl
import numpy as np
from tensorflow.python.pywrap_tensorflow import do_quantize_training_on_graphdef

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
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input
from keras.models import Model
from experience_memory import ExperienceMemory
from config_refiner import *
from sys import stdout

import tb_callback
import lrs_callback
import argparse
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
    # a1 = AttnLossLayer()(a1_reshape)
    dot_1 = multiply([out_1, a1_reshape])

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


def refiner_g_model():
    inputs = Input(shape=(10, 128, 128, 3))
    enc_inputs = Input(shape=(10, 128, 128, 32))
    conv_1 = TimeDistributed(Conv2D(filters=32,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding="same"))(inputs)
    conv_1 = TimeDistributed(BatchNormalization())(conv_1)
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

    # conv_4 = TimeDistributed(Conv2D(filters=128,
    #                                 kernel_size=(3, 3),
    #                                 strides=(2, 2),
    #                                 padding="same"))(enc_inputs)
    # conv_4 = TimeDistributed(BatchNormalization())(conv_4)
    # conv_4 = TimeDistributed(LeakyReLU(alpha=0.2))(conv_4)
    # conv_4 = TimeDistributed(Dropout(0.5))(conv_4)
    #
    # enc_2 = concatenate([conv_3, conv_4])

    # Decoder stage 2
    conv_5 = TimeDistributed(Conv2DTranspose(filters=128,
                                             kernel_size=(3, 3),
                                             strides=(1, 1),
                                             padding="same"))(conv_3)
    conv_5 = TimeDistributed(BatchNormalization())(conv_5)
    conv_5 = TimeDistributed(LeakyReLU(alpha=0.2))(conv_5)
    conv_5 = TimeDistributed(Dropout(0.5))(conv_5)

    conv_6 = TimeDistributed(Conv2DTranspose(filters=64,
                                             kernel_size=(3, 3),
                                             strides=(2, 2),
                                             padding="same"))(conv_5)
    conv_6 = TimeDistributed(BatchNormalization())(conv_6)
    conv_6 = TimeDistributed(LeakyReLU(alpha=0.2))(conv_6)
    conv_6 = TimeDistributed(Dropout(0.5))(conv_6)

    conv_7 = TimeDistributed(Conv2DTranspose(filters=32,
                                             kernel_size=(5, 5),
                                             strides=(2, 2),
                                             padding="same"))(conv_6)
    conv_7 = TimeDistributed(BatchNormalization())(conv_7)
    conv_7 = TimeDistributed(LeakyReLU(alpha=0.2))(conv_7)
    conv_7 = TimeDistributed(Dropout(0.5))(conv_7)

    conv_8 = TimeDistributed(Conv2DTranspose(filters=16,
                                             kernel_size=(5, 5),
                                             strides=(2, 2),
                                             padding="same"))(conv_7)
    conv_8 = TimeDistributed(BatchNormalization())(conv_8)
    conv_8 = TimeDistributed(LeakyReLU(alpha=0.2))(conv_8)
    conv_8 = TimeDistributed(Dropout(0.5))(conv_8)

    conv_9 = TimeDistributed(Conv2DTranspose(filters=3,
                                             kernel_size=(5, 5),
                                             strides=(2, 2),
                                             padding="same"))(conv_8)
    # conv_5 = TimeDistributed(BatchNormalization())(conv_5)
    conv_9 = TimeDistributed(LeakyReLU(alpha=0.2))(conv_9)
    conv_9 = TimeDistributed(Dropout(0.5))(conv_9)

    model = Model(inputs=inputs, outputs=conv_9)

    return model

def refiner_d_model():

    inputs = Input(shape=(10, 256, 256, 3))
    conv_1 = TimeDistributed(Conv2D(filters=32,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding="same"))(inputs)
    # conv_1 = TimeDistributed(BatchNormalization())(conv_1)
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

    conv_4 = TimeDistributed(Conv2D(filters=256,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding="same"))(conv_3)
    conv_4 = TimeDistributed(BatchNormalization())(conv_4)
    conv_4 = TimeDistributed(LeakyReLU(alpha=0.2))(conv_4)
    conv_4 = TimeDistributed(Dropout(0.5))(conv_4)

    flat_1 = TimeDistributed(Flatten())(conv_4)
    dense_1 = TimeDistributed(Dense(units=512, activation='tanh'))(flat_1)
    dense_1 = TimeDistributed(Dropout(0.5))(dense_1)
    dense_2 = TimeDistributed(Dense(units=1, activation='sigmoid'))(dense_1)

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


def gan_model(autoencoder, generator, discriminator):
    # model = Sequential()
    # set_trainability(autoencoder, False)
    # model.add(autoencoder)
    # model.add(generator)
    # set_trainability(discriminator, False)
    # model.add(discriminator)

    input = Input(shape=(int(VIDEO_LENGTH/2), 128, 128, 3))
    set_trainability(autoencoder, False)
    y128 = autoencoder(input)
    y256 = generator(y128)
    set_trainability(discriminator, False)
    reality = discriminator(y256)

    model = Model(inputs=input, outputs=[y256, reality])

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

def run_utilities(encoder, decoder, autoencoder, generator, discriminator, gan,
                  ENC_WEIGHTS, DEC_WEIGHTS, GEN_WEIGHTS, DIS_WEIGHTS):
# def run_utilities(encoder, decoder, autoencoder, ENC_WEIGHTS, DEC_WEIGHTS):
    if PRINT_MODEL_SUMMARY:
        print (encoder.summary())
        print (decoder.summary())
        print (autoencoder.summary())
        if ADVERSARIAL:
            print (generator.summary())
            print (discriminator.summary())
            print (gan.summary())
        # exit(0)

    # Save model to file
    if SAVE_MODEL:
        print ("Saving models to file...")
        model_json = encoder.to_json()
        with open(os.path.join(MODEL_DIR, "encoder.json"), "w") as json_file:
            json_file.write(model_json)

        model_json = decoder.to_json()
        with open(os.path.join(MODEL_DIR, "decoder.json"), "w") as json_file:
            json_file.write(model_json)

        model_json = autoencoder.to_json()
        with open(os.path.join(MODEL_DIR, "autoencoder.json"), "w") as json_file:
            json_file.write(model_json)

        if ADVERSARIAL:
            model_json = generator.to_json()
            with open(os.path.join(MODEL_DIR, "generator.json"), "w") as json_file:
                json_file.write(model_json)

            model_json = discriminator.to_json()
            with open(os.path.join(MODEL_DIR, "discriminator.json"), "w") as json_file:
                json_file.write(model_json)

            model_json = gan.to_json()
            with open(os.path.join(MODEL_DIR, "gan.json"), "w") as json_file:
                json_file.write(model_json)

        if PLOT_MODEL:
            plot_model(encoder, to_file=os.path.join(MODEL_DIR, 'encoder.png'), show_shapes=True)
            plot_model(decoder, to_file=os.path.join(MODEL_DIR, 'decoder.png'), show_shapes=True)
            plot_model(autoencoder, to_file=os.path.join(MODEL_DIR, 'autoencoder.png'), show_shapes=True)
            if ADVERSARIAL:
                plot_model(generator, to_file=os.path.join(MODEL_DIR, 'generator.png'), show_shapes=True)
                plot_model(discriminator, to_file=os.path.join(MODEL_DIR, 'discriminator.png'), show_shapes=True)
                plot_model(gan, to_file=os.path.join(MODEL_DIR, 'gan.png'), show_shapes=True)

    if ENC_WEIGHTS != "None":
        print ("Pre-loading encoder with weights...")
        load_weights(ENC_WEIGHTS, encoder)
    if DEC_WEIGHTS != "None":
        print ("Pre-loading decoder with weights...")
        load_weights(DEC_WEIGHTS, decoder)
    if ADVERSARIAL:
        if GEN_WEIGHTS != "None":
            print("Pre-loading generator with weights...")
            load_weights(GEN_WEIGHTS, generator)
        if DIS_WEIGHTS != "None":
            print("Pre-loading discriminator with weights...")
            load_weights(DIS_WEIGHTS, discriminator)


def load_X(videos_list, index, data_dir, img_size):
    X = np.zeros((BATCH_SIZE, VIDEO_LENGTH,) + img_size)
    for i in range(BATCH_SIZE):
        for j in range(VIDEO_LENGTH):
            filename = "frame_" + str(videos_list[(index*BATCH_SIZE + i), j]) + ".png"
            im_file = os.path.join(data_dir, filename)
            try:
                frame = cv2.imread(im_file, cv2.IMREAD_COLOR)
                X[i, j] = (frame.astype(np.float32) - 127.5) / 127.5
            except AttributeError as e:
                print (im_file)
                print (e)

    return X

def load_X_y(videos_list, index, frames, action_cats):
    # X = np.zeros((BATCH_SIZE, VIDEO_LENGTH,) + IMG_SIZE)
    X = []
    y = []
    for i in range(BATCH_SIZE):
        # y_per_vid = []
        start_index = videos_list[(index*BATCH_SIZE + i), 0]
        end_index = videos_list[(index*BATCH_SIZE + i), 1]
        X.append(frames[start_index:end_index])
        if (len(action_cats) != 0):
            y.append(action_cats[start_index:end_index])
        # for j in range(VIDEO_LENGTH):
        #     frame_number = (videos_list[(index*BATCH_SIZE + i), j])
        #     filename = "frame_" + str(frame_number) + ".png"
        #     im_file = os.path.join(data_dir, filename)
        #     try:
        #         frame = cv2.imread(im_file, cv2.IMREAD_COLOR)
        #         X[i, j] = (frame.astype(np.float32) - 127.5) / 127.5
        #     except AttributeError as e:
        #         print (im_file)
        #         print (e)
        #     if (len(action_cats) != 0):
        #         try:
        #             y_per_vid.append(action_cats[frame_number - 1])
        #         except IndexError as e:
        #             print (frame_number)
        #             print (e)
        # if (len(action_cats) != 0):
        #     y.append(y_per_vid)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y


def train(BATCH_SIZE, ENC_WEIGHTS, DEC_WEIGHTS, GEN_WEIGHTS, DIS_WEIGHTS):
    print ("Loading data definitions...")
    frames_source = hkl.load(os.path.join(DATA_DIR, 'sources_train_128.hkl'))

    # # Build video progressions
    # videos_list = []
    # start_frame_index = 1
    # end_frame_index = VIDEO_LENGTH + 1
    # while (end_frame_index <= len(frames_source)):
    #     frame_list = frames_source[start_frame_index:end_frame_index]
    #     if (len(set(frame_list)) == 1):
    #         videos_list.append(range(start_frame_index, end_frame_index))
    #         start_frame_index = start_frame_index + 1
    #         end_frame_index = end_frame_index + 1
    #     else:
    #         start_frame_index = end_frame_index - 1
    #         end_frame_index = start_frame_index + VIDEO_LENGTH
    #
    # videos_list = np.asarray(videos_list, dtype=np.int32)
    # n_videos = videos_list.shape[0]

    frames = np.zeros(shape=((len(frames_source),) + IMG_SIZE))

    j = 1
    for i in range(1, len(frames_source)):
        filename = "frame_" + str(j) + ".png"
        im_file = os.path.join(DATA_DIR, filename)
        try:
            frame = cv2.imread(im_file, cv2.IMREAD_COLOR)
            frames[i] = (frame.astype(np.float32) - 127.5) / 127.5
            # j = j + 2
            j = j + 1
        except AttributeError as e:
            print(im_file)
            print(e)

    # Build video progressions
    videos_list = []
    start_frame_index = 0
    end_frame_index = VIDEO_LENGTH
    while (end_frame_index <= len(frames_source)):
        frame_list = frames_source[start_frame_index:end_frame_index]
        if (len(set(frame_list)) == 1):
            videos_list.append((start_frame_index, end_frame_index))
            start_frame_index = start_frame_index + 1
            end_frame_index = end_frame_index + 1
        else:
            start_frame_index = end_frame_index - 1
            end_frame_index = start_frame_index + VIDEO_LENGTH

    videos_list = np.asarray(videos_list, dtype=np.int32)
    n_videos = videos_list.shape[0]

    # Setup validation
    val_frames_source = hkl.load(os.path.join(VAL_DATA_DIR, 'sources_val_128.hkl'))
    val_videos_list = []
    start_frame_index = 1
    end_frame_index = VIDEO_LENGTH + 1
    while (end_frame_index <= len(val_frames_source)):
        val_frame_list = val_frames_source[start_frame_index:end_frame_index]
        if (len(set(val_frame_list)) == 1):
            val_videos_list.append(range(start_frame_index, end_frame_index))
            start_frame_index = start_frame_index + VIDEO_LENGTH
            end_frame_index = end_frame_index + VIDEO_LENGTH
        else:
            start_frame_index = end_frame_index - 1
            end_frame_index = start_frame_index + VIDEO_LENGTH

    val_videos_list = np.asarray(val_videos_list, dtype=np.int32)
    n_val_videos = val_videos_list.shape[0]

    # if ADVERSARIAL:
    #     # Build video progressions
    #     print("Loading stage 2 data definitions...")
    #     frames_source_hd = hkl.load(os.path.join(DATA_DIR, 'sources_train_256.hkl'))
    #     videos_list_hd = []
    #     start_frame_index_hd = 1
    #     end_frame_index_hd = VIDEO_LENGTH + 1
    #     while (end_frame_index_hd <= len(frames_source_hd)):
    #         frame_list_hd = frames_source_hd[start_frame_index_hd:end_frame_index_hd]
    #         if (len(set(frame_list_hd)) == 1):
    #             videos_list_hd.append(range(start_frame_index_hd, end_frame_index_hd))
    #             start_frame_index_hd = start_frame_index_hd + 1
    #             end_frame_index = end_frame_index + 1
    #         else:
    #             start_frame_index = end_frame_index - 1
    #             end_frame_index = start_frame_index + VIDEO_LENGTH
    #
    #     videos_list = np.asarray(videos_list, dtype=np.int32)
    #     n_videos = videos_list.shape[0]

    if SHUFFLE:
        # Shuffle images to aid generalization
        videos_list = np.random.permutation(videos_list)

    # Build the Spatio-temporal Autoencoder
    print ("Creating models...")
    encoder = encoder_model()
    decoder = decoder_model()
    autoencoder = autoencoder_model(encoder, decoder)

    if ADVERSARIAL:
        generator = refiner_g_model()
        discriminator = refiner_d_model()
        gan = gan_model(autoencoder, generator, discriminator)
        generator.compile(loss='binary_crossentropy', optimizer='sgd')
        gan.compile(loss=['mae', 'binary_crossentropy'],
                    loss_weights=LOSS_WEIGHTS,
                    optimizer=OPTIM_G,
                    metrics=['accuracy'])
        set_trainability(discriminator, True)
        discriminator.compile(loss='binary_crossentropy',
                              optimizer=OPTIM_D,
                              metrics=['accuracy'])
        run_utilities(encoder, decoder, autoencoder, generator, discriminator, gan,
                      ENC_WEIGHTS, DEC_WEIGHTS, GEN_WEIGHTS, DIS_WEIGHTS)
    else:
        run_utilities(encoder, decoder, autoencoder, 'None', 'None', 'None',
                      ENC_WEIGHTS, DEC_WEIGHTS, 'None', 'None')

    autoencoder.compile(loss="mean_squared_error", optimizer=OPTIM_A)

    NB_ITERATIONS = int(n_videos/BATCH_SIZE)
    # NB_ITERATIONS = 1
    NB_VAL_ITERATIONS = int(n_val_videos / BATCH_SIZE)

    # Setup TensorBoard Callback
    TC = tb_callback.TensorBoard(log_dir=TF_LOG_DIR, histogram_freq=0, write_graph=False, write_images=False)
    TC_gan = tb_callback.TensorBoard(log_dir=TF_LOG_GAN_DIR, histogram_freq=0, write_graph=False, write_images=False)
    LRS = lrs_callback.LearningRateScheduler(schedule=schedule)
    LRS.set_model(autoencoder)

    print ("Beginning Training...")
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
            # X = load_X(videos_list, index, DATA_DIR, (128, 128, 3))
            X, y = load_X_y(videos_list, index, frames, [])
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
            X = load_X(val_videos_list, index, VAL_DATA_DIR, (128, 128, 3))
            X_train = X[:, 0: int(VIDEO_LENGTH / 2)]
            y_train = X[:, int(VIDEO_LENGTH / 2):]
            val_loss.append(autoencoder.test_on_batch(X_train, y_train))

            arrow = int(index / (NB_VAL_ITERATIONS / 40))
            stdout.write("\rIter: " + str(index) + "/" + str(NB_VAL_ITERATIONS - 1) + "  " +
                         "val_loss: " + str(val_loss[len(val_loss) - 1]) +
                         "\t    [" + "{0}>".format("=" * (arrow)))
            stdout.flush()

        # then after each epoch/iteration
        avg_loss = sum(loss)/len(loss)
        avg_val_loss = sum(val_loss) / len(val_loss)
        logs = {'loss': avg_loss, 'val_loss': avg_val_loss}
        TC.on_epoch_end(epoch, logs)

        # Log the losses
        with open(os.path.join(LOG_DIR, 'losses.json'), 'a') as log_file:
            log_file.write("{\"epoch\":%d, \"loss\":%f};\n" % (epoch, avg_loss))

            print("\nAvg loss: " + str(avg_loss) + " Avg val loss: " + str(avg_val_loss))

        # Save model weights per epoch to file
        encoder.save_weights(os.path.join(CHECKPOINT_DIR, 'encoder_epoch_' + str(epoch) + '.h5'), True)
        decoder.save_weights(os.path.join(CHECKPOINT_DIR, 'decoder_epoch_' + str(epoch) + '.h5'), True)

    # Train AAE
    if ADVERSARIAL:
        print ("Training Stage II.")
        exp_memory = ExperienceMemory(memory_length=100)
        for epoch in range(NB_EPOCHS_GAN):
            print("\n\nEpoch ", epoch)
            g_loss = []
            val_g_loss = []
            d_loss = []
            val_d_loss = []
            # a_loss = []

            # # Set learning rate every epoch
            # LRS.on_epoch_begin(epoch=epoch)
            lr = K.get_value(gan.optimizer.lr)
            print ("GAN learning rate: " + str(lr))
            lr = K.get_value(discriminator.optimizer.lr)
            print("Disc learning rate: " + str(lr))

            for index in range(NB_ITERATIONS):
                # Train Autoencoder
                X = load_X(videos_list, index, DATA_DIR, (128, 128, 3))
                X_hd = load_X(videos_list, index, HD_DATA_DIR, (256, 256, 3))
                X128 = X[:, 0 : int(VIDEO_LENGTH/2)]
                Y128 = autoencoder.predict(X128, verbose=0)
                X256_real = X_hd[:, int(VIDEO_LENGTH/2) :]
                X256_fake = generator.predict(Y128, verbose=0)

                trainable_fakes = exp_memory.get_trainable_fakes(current_gens=X256_fake, exp_window_size=4)

                # Train Discriminator on future images (y_train, not X_train)
                X = np.concatenate((X256_real, trainable_fakes))
                y = np.concatenate((np.ones(shape=(BATCH_SIZE, 10, 1), dtype=np.float32),
                                    np.zeros(shape=(BATCH_SIZE, 10, 1), dtype=np.float32)), axis=0)
                d_loss.append(discriminator.train_on_batch(X, y))

                # Train AAE
                set_trainability(discriminator, False)
                y = np.ones(shape=(BATCH_SIZE, 10, 1), dtype=np.float32)
                X128_fake = autoencoder.predict(X128, verbose=0)
                g_loss.append(gan.train_on_batch(X128_fake, [X256_real, y]))
                set_trainability(discriminator, True)

                # # Train Autoencoder
                # a_loss.append(autoencoder.train_on_batch(X_train, y_train))

                arrow = int(index / (NB_ITERATIONS / 30))
                stdout.write("\rIteration: " + str(index) + "/" + str(NB_ITERATIONS-1) + "  " +
                             "g_loss: " + str([g_loss[len(g_loss) - 1][j] for j in [0, -1]]) + "  " +
                             "d_loss: " + str(d_loss[len(d_loss) - 1]) +
                             "\t    [" + "{0}>".format("="*(arrow)))
                stdout.flush()

            if SAVE_GENERATED_IMAGES:
                # Save generated images to file
                predicted_images = generator.predict(X_train, verbose=0)
                orig_image, truth_image, pred_image = combine_images(X128_fake, X256_real, predicted_images)
                pred_image = pred_image * 127.5 + 127.5
                orig_image = orig_image * 127.5 + 127.5
                truth_image = truth_image * 127.5 + 127.5
                if epoch == 0 :
                    cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_gan_orig.png"), orig_image)
                    cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_gan_truth.png"), truth_image)
                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_gan_pred.png"), pred_image)

            # Run over validation data
            print('')
            for index in range(NB_VAL_ITERATIONS):
                X = load_X(val_videos_list, index, VAL_DATA_DIR, (128, 128, 3))
                X_hd = load_X(val_videos_list, index, VAL_HD_DATA_DIR, (256, 256, 3))
                X128_val = X[:, 0: int(VIDEO_LENGTH / 2)]
                Y128_val = autoencoder.predict(X128, verbose=0)
                X256_real_val = X_hd[:, int(VIDEO_LENGTH / 2):]
                X256_fake_val = generator.predict(Y128_val, verbose=0)

                X = np.concatenate((X256_real_val, X256_fake_val))
                y = np.concatenate((np.ones(shape=(BATCH_SIZE, 10, 1), dtype=np.float32),
                                    np.zeros(shape=(BATCH_SIZE, 10, 1), dtype=np.float32)), axis=0)
                val_d_loss.append(discriminator.test_on_batch(X256_fake_val, X256_real_val))

                y = np.ones(shape=(BATCH_SIZE, 10, 1), dtype=np.float32)
                val_g_loss.append(gan.test_on_batch(X128_val, [X256_real_val, y]))

                arrow = int(index / (NB_VAL_ITERATIONS / 40))
                stdout.write("\rIter: " + str(index) + "/" + str(NB_VAL_ITERATIONS - 1) + "  " +
                             "val_g_loss: " + str([val_g_loss[len(val_g_loss) - 1][j] for j in [0, -1]]) + "  " +
                             "val_d_loss: " + str(val_d_loss[len(val_d_loss) - 1]))
                stdout.flush()

            # then after each epoch/iteration
            avg_d_loss = np.mean(np.asarray(d_loss, dtype=np.float32), axis=0)
            avg_val_d_loss = np.mean(np.asarray(val_d_loss, dtype=np.float32), axis=0)
            avg_g_loss = np.mean(np.asarray(g_loss, dtype=np.float32), axis=0)
            avg_val_g_loss = np.mean(np.asarray(val_g_loss, dtype=np.float32), axis=0)

            loss_values = np.asarray(avg_d_loss.tolist() + avg_val_d_loss.tolist() \
                                     + avg_g_loss.tolist() + avg_val_g_loss.tolist(), dtype=np.float32)
            d_loss_keys = ['d_' + metric for metric in discriminator.metrics_names]
            g_loss_keys = ['g_' + metric for metric in gan.metrics_names]
            val_d_loss_keys = ['d_val_' + metric for metric in discriminator.metrics_names]
            val_g_loss_keys = ['g_val_' + metric for metric in gan.metrics_names]

            loss_keys = d_loss_keys + val_d_loss_keys + \
                        g_loss_keys + val_g_loss_keys
            logs = dict(zip(loss_keys, loss_values))

            TC_gan.on_epoch_end(epoch, logs)

            # Log the losses
            with open(os.path.join(LOG_DIR, 'losses_gan.json'), 'a') as log_file:
                log_file.write("{\"epoch\":%d, %s;\n" % (epoch, logs))

            print("\nAvg d_loss: " + str(avg_d_loss) +
                  " Avg val_d_loss: " + str(avg_val_d_loss) +
                  "\nAvg g_loss: " + str([avg_g_loss[len(avg_g_loss) - 1][j] for j in [0, -1]]) +
                  " Avg val_g_loss: " + str([avg_val_g_loss[len(avg_val_g_loss) - 1][j] for j in [0, -1]]))

            # Save model weights per epoch to file
            encoder.save_weights(os.path.join(CHECKPOINT_DIR, 'encoder_gan_epoch_'+str(epoch)+'.h5'), True)
            decoder.save_weights(os.path.join(CHECKPOINT_DIR, 'decoder_gan_epoch_' + str(epoch) + '.h5'), True)
            generator.save_weights(os.path.join(CHECKPOINT_DIR,
                                                    'generator_gan_epoch_' +
                                                    str(epoch) + '.h5'), True)
            discriminator.save_weights(os.path.join(CHECKPOINT_DIR,
                                                    'discriminator_gan_epoch_' +
                                                    str(epoch) + '.h5'), True)

    # End TensorBoard Callback
    TC.on_train_end('_')


def test(ENC_WEIGHTS, DEC_WEIGHTS):

    # Create models
    print ("Creating models...")
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
    parser.add_argument("--gen_weights", type=str, default="None")
    parser.add_argument("--dis_weights", type=str, default="None")
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
              GEN_WEIGHTS=args.gen_weights,
              DIS_WEIGHTS=args.dis_weights)

    if args.mode == "test":
        test(ENC_WEIGHTS=args.enc_weights,
             DEC_WEIGHTS=args.dec_weights)
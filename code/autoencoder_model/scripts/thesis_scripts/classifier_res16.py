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
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import UpSampling3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.merge import multiply
from keras.layers.merge import add
from keras.layers.merge import concatenate
from keras.layers.core import Dense
from keras.layers.core import Lambda
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input
from keras.models import Model
from keras.models import model_from_json
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from image_utils import random_rotation
from image_utils import random_shift
from image_utils import flip_axis
from image_utils import random_brightness
from config_clar16 import *
from sys import stdout

import tb_callback
import lrs_callback
import argparse
import random
import math
import cv2
import os


def encoder_model():
    inputs = Input(shape=(int(VIDEO_LENGTH/2), 128, 208, 3))

    # 10x128x128
    conv_1 = Conv3D(filters=128,
                     strides=(1, 4, 4),
                     dilation_rate=(1, 1, 1),
                     kernel_size=(3, 11, 11),
                     padding='same')(inputs)
    x = TimeDistributed(BatchNormalization())(conv_1)
    x = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_1 = TimeDistributed(Dropout(0.5))(x)

    conv_2a = Conv3D(filters=64,
                     strides=(1, 1, 1),
                     dilation_rate=(2, 1, 1),
                     kernel_size=(2, 5, 5),
                     padding='same')(out_1)
    x = TimeDistributed(BatchNormalization())(conv_2a)
    x = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_2a = TimeDistributed(Dropout(0.5))(x)

    conv_2b = Conv3D(filters=64,
                    strides=(1, 1, 1),
                    dilation_rate=(2, 1, 1),
                    kernel_size=(2, 5, 5),
                    padding='same')(out_2a)
    x = TimeDistributed(BatchNormalization())(conv_2b)
    x = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_2b = TimeDistributed(Dropout(0.5))(x)

    conv_2c = TimeDistributed(Conv2D(filters=64,
                                        kernel_size=(1, 1),
                                        strides=(1, 1),
                                        padding='same'))(out_1)
    x = TimeDistributed(BatchNormalization())(conv_2c)
    out_1_less = TimeDistributed(LeakyReLU(alpha=0.2))(x)

    res_1 = add([out_1_less, out_2b])
    # res_1 = LeakyReLU(alpha=0.2)(res_1)

    conv_3 = Conv3D(filters=64,
                     strides=(1, 2, 2),
                     dilation_rate=(1, 1, 1),
                     kernel_size=(3, 5, 5),
                     padding='same')(res_1)
    x = TimeDistributed(BatchNormalization())(conv_3)
    x = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_3 = TimeDistributed(Dropout(0.5))(x)

    # 10x16x16
    conv_4a = Conv3D(filters=64,
                     strides=(1, 1, 1),
                     dilation_rate=(2, 1, 1),
                     kernel_size=(2, 3, 3),
                     padding='same')(out_3)
    x = TimeDistributed(BatchNormalization())(conv_4a)
    x = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_4a = TimeDistributed(Dropout(0.5))(x)

    conv_4b = Conv3D(filters=64,
                     strides=(1, 1, 1),
                     dilation_rate=(2, 1, 1),
                     kernel_size=(2, 3, 3),
                     padding='same')(out_4a)
    x = TimeDistributed(BatchNormalization())(conv_4b)
    x = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_4b = TimeDistributed(Dropout(0.5))(x)

    z = add([out_3, out_4b])
    # res_1 = LeakyReLU(alpha=0.2)(res_1)

    model = Model(inputs=inputs, outputs=z)

    return model


def decoder_model():
    inputs = Input(shape=(int(VIDEO_LENGTH/2), 16, 26, 64))

    # 10x16x16
    convlstm_1 = ConvLSTM2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(inputs)
    x = TimeDistributed(BatchNormalization())(convlstm_1)
    out_1 = TimeDistributed(Activation('tanh'))(x)

    convlstm_2 = ConvLSTM2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(out_1)
    x = TimeDistributed(BatchNormalization())(convlstm_2)
    out_2 = TimeDistributed(Activation('tanh'))(x)

    # conv_1c = TimeDistributed(Conv2D(filters=64,
    #                                  kernel_size=(1, 1),
    #                                  strides=(1, 1),
    #                                  padding='same'))(inputs)
    # x = TimeDistributed(BatchNormalization())(conv_1c)
    # res_0_less = TimeDistributed(Activation('tanh'))(x)
    res_1 = add([inputs, out_2])
    res_1 = UpSampling3D(size=(1, 2, 2))(res_1)

    # 10x32x32
    convlstm_3a = ConvLSTM2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(res_1)
    x = TimeDistributed(BatchNormalization())(convlstm_3a)
    out_3a = TimeDistributed(Activation('tanh'))(x)

    convlstm_3b = ConvLSTM2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(out_3a)
    x = TimeDistributed(BatchNormalization())(convlstm_3b)
    out_3b = TimeDistributed(Activation('tanh'))(x)

    # conv_3c = TimeDistributed(Conv2D(filters=64,
    #                                     kernel_size=(1, 1),
    #                                     strides=(1, 1),
    #                                     padding='same'))(res_1)
    # x = TimeDistributed(BatchNormalization())(conv_3c)
    # res_1_less = TimeDistributed(Activation('tanh'))(x)
    res_2 = add([res_1, out_3b])
    res_2 = UpSampling3D(size=(1, 2, 2))(res_2)

    # 10x64x64
    convlstm_4a = ConvLSTM2D(filters=16,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(res_2)
    x = TimeDistributed(BatchNormalization())(convlstm_4a)
    out_4a = TimeDistributed(Activation('tanh'))(x)

    convlstm_4b = ConvLSTM2D(filters=16,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(out_4a)
    x = TimeDistributed(BatchNormalization())(convlstm_4b)
    out_4b = TimeDistributed(Activation('tanh'))(x)

    conv_4c = TimeDistributed(Conv2D(filters=16,
                                        kernel_size=(1, 1),
                                        strides=(1, 1),
                                        padding='same'))(res_2)
    x = TimeDistributed(BatchNormalization())(conv_4c)
    res_2_less = TimeDistributed(Activation('tanh'))(x)
    res_3 = add([res_2_less, out_4b])
    res_3 = UpSampling3D(size=(1, 2, 2))(res_3)

    # 10x128x128
    convlstm_5 = ConvLSTM2D(filters=3,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(res_3)
    predictions = TimeDistributed(Activation('tanh'))(convlstm_5)

    model = Model(inputs=inputs, outputs=predictions)

    return model


def process_prec3d():
    json_file = open(PRETRAINED_C3D, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    model.load_weights(PRETRAINED_C3D_WEIGHTS)
    print("Loaded weights from disk")
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

    return model


def pretrained_c3d():
    c3d = process_prec3d()
    inputs = Input(shape=(16, 128, 112, 3))
    resized = TimeDistributed(Lambda(lambda image: tf.image.resize_images(image, (112, 112))))(inputs)
    c3d_out = c3d(resized)
    model = Model(inputs=inputs, outputs=c3d_out)
    # print (c3d.summary())

    return model


def ensemble_c3d():
    inputs = Input(shape=(16, 128, 208, 3))

    def sliceA(x):
        return x[:, :, :, 0:112, :]

    def sliceB(x):
        return x[:, :, :, 96:208, :]

    A = Lambda(sliceA)(inputs)
    B = Lambda(sliceB)(inputs)

    c3d_A = pretrained_c3d()
    c3d_B = pretrained_c3d()

    A_out = c3d_A(A)
    B_out = c3d_B(B)

    features = concatenate([A_out, B_out])
    dense = Dense(units=1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(features)
    x = BatchNormalization()(dense)
    x = Dropout(0.5)(x)
    x = Dense(units=512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    actions = Dense(units=len(simple_ped_set), activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(x)

    model = Model(inputs=inputs, outputs=actions)

    return model


def load_to_RAM(frames_source):
    frames = np.zeros(shape=((len(frames_source),) + IMG_SIZE))

    j = 1
    for i in range(1, len(frames_source)):
        filename = "frame_" + str(j) + ".png"
        im_file = os.path.join(DATA_DIR, filename)
        try:
            frame = cv2.imread(im_file, cv2.IMREAD_COLOR)
            frames[i] = frame.astype(np.float32)
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
    input = Input(shape=(16, 128, 208, 3))
    set_trainability(encoder, RETRAIN_GENERATOR)
    z = encoder(input)
    set_trainability(decoder, RETRAIN_GENERATOR)
    future = decoder(z)
    set_trainability(classifier, RETRAIN_CLASSIFIER)
    actions = classifier(future)

    model = Model(inputs=input, outputs=[future, actions])

    return model


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
    width = img_width * video_stack.shape[1]
    # height = img_size x batch_size
    height = img_height * video_stack.shape[0]
    shape = frames.shape[1:]
    image = np.zeros((height, width, shape[2]), dtype=video_stack.dtype)
    frame_number = 0

    for i in range(video_stack.shape[0]):
        for j in range(video_stack.shape[1]):
            image[(i * img_height):((i + 1) * img_height), (j * img_width):((j + 1) * img_width)] = frames[frame_number]
            frame_number = frame_number + 1

    return image



def load_weights(weights_file, model):
    model.load_weights(weights_file)


def run_utilities(encoder, decoder, classifier, ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS):
    if PRINT_MODEL_SUMMARY:
        print ("Encoder:")
        print (encoder.summary())
        print ("Decoder:")
        print (decoder.summary())
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

        if CLASSIFIER:
            model_json = classifier.to_json()
            with open(os.path.join(MODEL_DIR, "classifier.json"), "w") as json_file:
                json_file.write(model_json)

        if PLOT_MODEL:
            plot_model(encoder, to_file=os.path.join(MODEL_DIR, 'encoder.png'), show_shapes=True)
            plot_model(decoder, to_file=os.path.join(MODEL_DIR, 'decoder.png'), show_shapes=True)
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


def random_augmentation(video):
    # Toss a die
    if RANDOM_AUGMENTATION:
        k = np.random.randint(0, 5, dtype=int)
    else:
        k = 0
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
            video = np.take(frames, videos_list[(index*BATCH_SIZE + i)], axis=0)
            video = random_augmentation(video)
            X.append(video)
            if (len(ped_action_cats) != 0):
                y.append(np.take(ped_action_cats, videos_list[(index * BATCH_SIZE + i)], axis=0))

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
            if action.lower() == 'crossing':
                ped_action = simple_ped_set.index('crossing')
                simple_ped_actions_per_frame.append(ped_action)

        for action in simple_ped_actions_per_frame:
            count[action] = count[action] + 1
            # Add all unique categorical one-hot vectors
            encoded_ped_action = encoded_ped_action + to_categorical(action, len(simple_ped_set))

        ped_action_classes.append(encoded_ped_action.T)

    ped_action_classes = np.asarray(ped_action_classes)
    ped_action_classes = np.reshape(ped_action_classes, newshape=(ped_action_classes.shape[0:2]))
    return ped_action_classes, count


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


def get_sklearn_metrics(y_true, y_pred, avg=None):
    return precision_recall_fscore_support(y_true, np.round(y_pred), average=avg)


def get_classification_report(y_true, y_pred):
    return classification_report(y_true, np.round(y_pred), target_names=simple_ped_set)


def train(BATCH_SIZE, ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS):
    print("Loading data definitions.")

    frames_source = hkl.load(os.path.join(DATA_DIR, 'sources_train_208.hkl'))
    videos_list_1 = get_video_lists(frames_source=frames_source, stride=8, frame_skip=0)
    videos_list_2 = get_video_lists(frames_source=frames_source, stride=8, frame_skip=1)
    videos_list = np.concatenate((videos_list_1, videos_list_2), axis=0)

    # Load actions from annotations
    action_labels = hkl.load(os.path.join(DATA_DIR, 'annotations_train_208.hkl'))
    ped_action_classes, ped_class_count = get_action_classes(action_labels=action_labels)
    print("Training Stats: " + str(ped_class_count))

    if RAM_DECIMATE:
        frames = load_to_RAM(frames_source=frames_source)

    if SHUFFLE:
        # Shuffle images to aid generalization
        videos_list = np.random.permutation(videos_list)

    # Setup test
    val_frames_source = hkl.load(os.path.join(VAL_DATA_DIR, 'sources_val_208.hkl'))
    val_videos_list = get_video_lists(frames_source=val_frames_source, stride=8, frame_skip=0)

    # Load test action annotations
    val_action_labels = hkl.load(os.path.join(VAL_DATA_DIR, 'annotations_test_208.hkl'))
    val_ped_action_classes, val_ped_class_count = get_action_classes(val_action_labels)
    print("Test Stats: " + str(val_ped_class_count))

    # Build the Spatio-temporal Autoencoder
    print ("Creating models.")
    encoder = encoder_model()
    decoder = decoder_model()

    # Build stacked classifier
    if CLASSIFIER:
        classifier = ensemble_c3d()
        run_utilities(encoder, decoder, classifier, ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS)
        sclassifier = stacked_classifier_model(encoder, decoder, classifier)
        sclassifier.compile(loss=["mae", "binary_crossentropy"],
                            loss_weights=LOSS_WEIGHTS,
                            optimizer=OPTIM_C,
                            metrics=['accuracy'])
        print (sclassifier.summary())

    if not CLASSIFIER:
        autoencoder = autoencoder_model(encoder, decoder)
        autoencoder.compile(loss="mean_absolute_error", optimizer=OPTIM_A)
        run_utilities(encoder, decoder, 'classifier', ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS)

    n_videos = videos_list.shape[0]
    n_test_videos = val_videos_list.shape[0]
    NB_ITERATIONS = int(n_videos/BATCH_SIZE)
    # NB_ITERATIONS = 1
    NB_VAL_ITERATIONS = int(n_test_videos/BATCH_SIZE)
    # NB_VAL_ITERATIONS = 1

    # Setup TensorBoard Callback
    TC = tb_callback.TensorBoard(log_dir=TF_LOG_DIR, histogram_freq=0, write_graph=False, write_images=False)
    TC_cla = tb_callback.TensorBoard(log_dir=TF_LOG_CLA_DIR, histogram_freq=0, write_graph=False, write_images=False)
    if not CLASSIFIER:
        LRS_auto = lrs_callback.LearningRateScheduler(schedule=auto_schedule)
        LRS_auto.set_model(autoencoder)

    if CLASSIFIER:
        LRS_cla = lrs_callback.LearningRateScheduler(schedule=cla_schedule)
        LRS_cla.set_model(sclassifier)

    print ("Beginning Training.")
    # Begin Training

    for epoch in range(1, NB_EPOCHS_AUTOENCODER+1):
        if epoch == 21:
            autoencoder.compile(loss="mean_absolute_error", optimizer=OPTIM_B)
            load_weights(os.path.join(CHECKPOINT_DIR, 'encoder_epoch_20.h5'), encoder)
            load_weights(os.path.join(CHECKPOINT_DIR, 'decoder_epoch_20.h5'), decoder)

        print("\n\nEpoch ", epoch)
        loss = []
        val_loss = []

        # Set learning rate every epoch
        LRS_auto.on_epoch_begin(epoch=epoch)
        lr = K.get_value(autoencoder.optimizer.lr)
        print("Learning rate: " + str(lr))

        for index in range(NB_ITERATIONS):
            # Train Autoencoder
            if RAM_DECIMATE:
                X, y = load_X_y_RAM(videos_list, index, frames, [])
            else:
                X, y = load_X_y(videos_list, index, DATA_DIR, [])

            X_train = np.flip(X[:, 0: int(VIDEO_LENGTH / 2)], axis=1)
            y_train = X[:, int(VIDEO_LENGTH / 2):]
            loss.append(autoencoder.train_on_batch(X_train, y_train))

            arrow = int(index / (NB_ITERATIONS / 40))
            stdout.write("\rIter: " + str(index) + "/" + str(NB_ITERATIONS - 1) + "  " +
                         "loss: " + str(loss[len(loss) - 1]) +
                         "\t    [" + "{0}>".format("=" * (arrow)))
            stdout.flush()

        if SAVE_GENERATED_IMAGES:
            # Save generated images to file
            predicted_images = autoencoder.predict(X_train, verbose=0)
            voila = np.concatenate((X_train, y_train), axis=1)
            truth_seq = arrange_images(voila)
            pred_seq = arrange_images(np.concatenate((X_train, predicted_images), axis=1))

            truth_seq = truth_seq * 127.5 + 127.5
            pred_seq = pred_seq * 127.5 + 127.5

            if epoch == 1:
                cv2.imwrite(os.path.join(GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_truth.png"), truth_seq)
            cv2.imwrite(os.path.join(GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_pred.png"), pred_seq)

        # Run over test data
        print('')
        for index in range(NB_VAL_ITERATIONS):
            X, y = load_X_y(val_videos_list, index, VAL_DATA_DIR, [])
            X_val = np.flip(X[:, 0: int(VIDEO_LENGTH / 2)], axis=1)
            y_val = X[:, int(VIDEO_LENGTH / 2):]
            val_loss.append(autoencoder.test_on_batch(X_val, y_val))

            arrow = int(index / (NB_VAL_ITERATIONS / 40))
            stdout.write("\rIter: " + str(index) + "/" + str(NB_VAL_ITERATIONS - 1) + "  " +
                         "val_loss: " + str(val_loss[len(val_loss) - 1]) +
                         "\t    [" + "{0}>".format("=" * (arrow)))
            stdout.flush()

        # then after each epoch/iteration
        avg_loss = sum(loss) / len(loss)
        avg_val_loss = sum(val_loss) / len(val_loss)
        logs = {'loss': avg_loss, 'val_loss': avg_val_loss}
        TC.on_epoch_end(epoch, logs)

        # Log the losses
        with open(os.path.join(LOG_DIR, 'losses_gen.json'), 'a') as log_file:
            log_file.write("{\"epoch\":%d, \"train_loss\":%f, \"val_loss\":%f}\n" % (epoch, avg_loss, avg_val_loss))

            print("\nAvg train loss: " + str(avg_loss) + " Avg val loss: " + str(avg_val_loss))

        # Save model weights per epoch to file
        if epoch > 15 and epoch < 21:
            encoder.save_weights(os.path.join(CHECKPOINT_DIR, 'encoder_epoch_' + str(epoch) + '.h5'), True)
            decoder.save_weights(os.path.join(CHECKPOINT_DIR, 'decoder_epoch_' + str(epoch) + '.h5'), True)
        if epoch > 25:
            encoder.save_weights(os.path.join(CHECKPOINT_DIR, 'encoder_epoch_' + str(epoch) + '.h5'), True)
            decoder.save_weights(os.path.join(CHECKPOINT_DIR, 'decoder_epoch_' + str(epoch) + '.h5'), True)

    # Train Classifier
    if CLASSIFIER:
        print("Training Classifier...")
        for epoch in range(1, NB_EPOCHS_CLASS+1):
            print("\n\nEpoch ", epoch)
            c_loss = []
            val_c_loss = []

            # # Set learning rate every epoch
            LRS_cla.on_epoch_begin(epoch=epoch)
            lr = K.get_value(sclassifier.optimizer.lr)

            y_train_pred = []
            y_train_true = []
            print("Learning rate: " + str(lr))
            print("c_loss_metrics: " + str(sclassifier.metrics_names))

            for index in range(NB_ITERATIONS):
                if RAM_DECIMATE:
                    X, y = load_X_y_RAM(videos_list, index, frames, ped_action_classes)
                else:
                    X, y = load_X_y(videos_list, index, DATA_DIR, ped_action_classes)

                X_train = np.flip(X[:, 0: int(VIDEO_LENGTH / 2)], axis=1)
                y_true_class = y[:, CLASS_TARGET_INDEX]
                y_true_imgs = X[:, int(VIDEO_LENGTH / 2):]

                c_loss.append(sclassifier.train_on_batch(X_train, [y_true_imgs, y_true_class]))

                y_train_true.extend(y_true_class)
                y_train_pred.extend(sclassifier.predict(X_train, verbose=0))

                arrow = int(index / (NB_ITERATIONS / 30))
                stdout.write("\rIter: " + str(index) + "/" + str(NB_ITERATIONS - 1) + "  " +
                             "c_loss: " + str([ c_loss[len(c_loss) - 1][j]  for j in [0, 1]]) + "  " +
                             "\t    [" + "{0}>".format("=" * (arrow)))
                stdout.flush()

            if SAVE_GENERATED_IMAGES:
                # Save generated images to file
                # generator = autoencoder_model(encoder, decoder)
                # predicted_images = generator.predict(X_train)
                predicted_images, ped_pred_class = sclassifier.predict(X_train, verbose=0)
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
                        # class_num_past = np.argmax(y_orig_classes[k, j])
                        # class_num_futr = np.argmax(y_true_classes[k, j])
                        # class_num_y = np.argmax(ped_pred_class[k])
                        # label_true = simple_ped_set[class_num_futr]
                        # label_orig = simple_ped_set[class_num_past]
                        # label_pred = simple_ped_set[class_num_y]
                        #
                        # label_true = str(y_orig_classes[k, j])
                        # label_pred = str([round(float(i), 2) for i in ped_pred_class[k]])

                        if (y_orig_classes[k, j] > 0.5):
                            label_orig = "crossing"
                        else:
                            label_orig = "not crossing"

                        if (y_true_classes[k][0] > 0.5):
                            label_true = "crossing"
                        else:
                            label_true = "not crossing"

                        if (ped_pred_class[k][0] > 0.5):
                            label_pred = "crossing"
                        else:
                            label_pred = "not crossing"

                        cv2.putText(pred_seq, label_orig,
                                    (2 + j * (208), 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        cv2.putText(pred_seq, label_pred,
                                    (2 + (j + 16) * (208), 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        cv2.putText(pred_seq, 'truth: ' + label_true,
                                    (2 + (j + 16) * (208), 94 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        cv2.putText(truth_image, label_true,
                                    (2 + j * (208), 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)

                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_cla_pred.png"), pred_seq)
                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_cla_truth.png"), truth_image)

            # Run over test data
            print('')
            y_val_pred = []
            y_val_true = []
            for index in range(NB_VAL_ITERATIONS):
                X, y = load_X_y(val_videos_list, index, VAL_DATA_DIR, val_ped_action_classes)
                X_val = X[:, 0: int(VIDEO_LENGTH / 2)]
                y_true_class = y[:, CLASS_TARGET_INDEX]
                y_true_imgs = X[:, int(VIDEO_LENGTH / 2):]

                val_c_loss.append(sclassifier.test_on_batch(X_val, [y_true_imgs, y_true_class]))
                y_val_true.extend(y_true_class)
                y_val_pred.extend(sclassifier.predict(X_val, verbose=0))

                arrow = int(index / (NB_VAL_ITERATIONS / 40))
                stdout.write("\rIter: " + str(index) + "/" + str(NB_VAL_ITERATIONS - 1) + "  " +
                             "val_c_loss: " +  str([ val_c_loss[len(val_c_loss) - 1][j]  for j in [0, 1]]))
                stdout.flush()

            # Save generated images to file
            # generator = autoencoder_model(encoder, decoder)
            # test_predicted_images = generator.predict(X_test)
            val_predicted_images, val_ped_pred_class = sclassifier.predict(X_val, verbose=0)
            orig_image = arrange_images(X_val)
            truth_image = arrange_images(y_true_imgs)
            pred_image = arrange_images(val_predicted_images)
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
                        # class_num_past = np.argmax(y_orig_classes[k, j])
                        # class_num_futr = np.argmax(y_true_classes[k, j])
                        if (y_orig_classes[k, j] > 0.5):
                            label_orig = "crossing"
                        else:
                            label_orig = "not crossing"

                        if (y_true_classes[k][0] > 0.5):
                            label_true = "crossing"
                        else:
                            label_true = "not crossing"

                        cv2.putText(orig_image, label_orig,
                                    (2 + j * (208), 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        cv2.putText(truth_image, label_true,
                                    (2 + j * (208), 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) +
                                         "_cla_test_orig.png"), orig_image)
                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) +
                                         "_cla_test_truth.png"), truth_image)

            # Add labels as text to the image
            for k in range(BATCH_SIZE):
                # class_num_y = np.argmax(test_ped_pred_class[k])
                if (val_ped_pred_class[k][0] > 0.5):
                    label_pred = "crossing"
                else:
                    label_pred = "not crossing"

                for j in range(int(VIDEO_LENGTH / 2)):
                    cv2.putText(pred_image, label_pred,
                                (2 + j * (208), 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                                cv2.LINE_AA)
            cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_cla_val_pred.png"),
                        pred_image)

            # then after each epoch/iteration
            avg_c_loss = np.mean(np.asarray(c_loss, dtype=np.float32), axis=0)
            avg_val_c_loss = np.mean(np.asarray(val_c_loss, dtype=np.float32), axis=0)

            # Calculate Precision and Recall scores
            train_prec, train_rec, train_fbeta, train_support = get_sklearn_metrics(np.asarray(y_train_true),
                                                                                    np.asarray(y_train_pred),
                                                                                    avg='micro')
            val_prec, val_rec, val_fbeta, val_support = get_sklearn_metrics(np.asarray(y_val_true),
                                                                                np.asarray(y_val_pred),
                                                                                avg='micro')
            print("Train Prec: %.2f, Recall: %.2f, Fbeta: %.2f" % (train_prec, train_rec, train_fbeta))
            print("Val Prec: %.2f, Recall: %.2f, Fbeta: %.2f" % (val_prec, val_rec, val_fbeta))
            loss_values = np.asarray(avg_c_loss.tolist() + [train_prec.tolist()] +
                                     [train_rec.tolist()] +
                                     avg_val_c_loss.tolist() + [val_prec.tolist()] +
                                     [val_rec.tolist()], dtype=np.float32)
            # loss_values = np.asarray(avg_c_loss.tolist() + train_prec.tolist() +
            #                          train_rec.tolist() +
            #                          avg_test_c_loss.tolist() + test_prec.tolist() +
            #                          test_rec.tolist(), dtype=np.float32)
            precs = ['prec_' + action for action in simple_ped_set]
            recs = ['rec_' + action for action in simple_ped_set]
            c_loss_keys = ['c_' + metric for metric in sclassifier.metrics_names + precs + recs]
            test_c_loss_keys = ['c_val_' + metric for metric in sclassifier.metrics_names + precs + recs]

            loss_keys = c_loss_keys + test_c_loss_keys
            logs = dict(zip(loss_keys, loss_values))

            TC_cla.on_epoch_end(epoch, logs)

            # Log the losses
            with open(os.path.join(LOG_DIR, 'losses_cla.json'), 'a') as log_file:
                log_file.write("{\"epoch\":%d, %s;\n" % (epoch, logs))

            print("\nAvg c_loss: " + str(avg_c_loss) +
                  " Avg test_c_loss: " + str(avg_val_c_loss))

            # prec, recall, fbeta, support = get_sklearn_metrics(np.asarray(y_train_true),
            #                                                    np.asarray(y_train_pred),
            #                                                    avg='micro')
            #
            # prec, recall, fbeta, support = get_sklearn_metrics(np.asarray(y_val_true),
            #                                                    np.asarray(y_val_pred),
            #                                                    avg='micro')
            #

            # Save model weights per epoch to file
            encoder.save_weights(os.path.join(CHECKPOINT_DIR, 'encoder_cla_epoch_' + str(epoch) + '.h5'), True)
            decoder.save_weights(os.path.join(CHECKPOINT_DIR, 'decoder_cla_epoch_' + str(epoch) + '.h5'), True)
            classifier.save_weights(os.path.join(CHECKPOINT_DIR, 'classifier_cla_epoch_' + str(epoch) + '.h5'),
                                    True)
        print(get_classification_report(np.asarray(y_train_true), np.asarray(y_train_pred)))
        print(get_classification_report(np.asarray(y_val_true), np.asarray(y_val_pred)))



def test(ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS):

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
        LAMBDA = 0
        train(BATCH_SIZE=args.batch_size,
              ENC_WEIGHTS=args.enc_weights,
              DEC_WEIGHTS=args.dec_weights,
              CLA_WEIGHTS=args.cla_weights)

    if args.mode == "test":
        test(ENC_WEIGHTS=args.enc_weights,
             DEC_WEIGHTS=args.dec_weights,
             CLA_WEIGHTS=args.cla_weights)

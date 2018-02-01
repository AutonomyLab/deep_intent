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
                     input_shape=(int(VIDEO_LENGTH/2), 112, 112, 3)))
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
    inputs = Input(shape=(16, 14, 14, 64))

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
    aclstm_1 = GRU(units=14 * 14,
                   activation='tanh',
                   recurrent_dropout=0.5,
                   return_sequences=True)(flat_1)
    x = TimeDistributed(BatchNormalization())(aclstm_1)
    dense_1 = TimeDistributed(Dense(units=14 * 14, activation='softmax'))(x)
    a1_reshape = Reshape(target_shape=(16, 14, 14, 1))(dense_1)
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


def process_prec3d():
    json_file = open(PRETRAINED_C3D, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    model.load_weights(PRETRAINED_C3D_WEIGHTS)
    print("Loaded model from disk")
    for layer in model.layers[:13]:
        layer.trainable = False

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

    # model.layers.pop()
    # model.outputs = [model.layers[-1].output]
    # model.layers[-1].outbound_nodes = []
    #
    # model.layers.pop()
    # model.outputs = [model.layers[-1].output]
    # model.layers[-1].outbound_nodes = []
    #
    # model.layers.pop()
    # model.outputs = [model.layers[-1].output]
    # model.layers[-1].outbound_nodes = []
    #
    # model.layers.pop()
    # model.outputs = [model.layers[-1].output]
    # model.layers[-1].outbound_nodes = []
    #
    # model.layers.pop()
    # model.outputs = [model.layers[-1].output]
    # model.layers[-1].outbound_nodes = []
    #
    # model.layers.pop()
    # model.outputs = [model.layers[-1].output]
    # model.layers[-1].outbound_nodes = []
    #
    # i=0
    # for layer in model.layers:
    #     print (layer, i)
    #     i = i+1
    # # exit(0)

    return model


def pretrained_c3d():
    c3d = process_prec3d()

    inputs = Input(shape=(16, 112, 112, 3))
    c3d_out = c3d(inputs)

    print (c3d.summary())

    # flat = TimeDistributed(Flatten())(c3d_out)

    # gru1 = GRU(units=256,
    #            return_sequences=True,
    #            recurrent_dropout=0.5)(flat)
    # x = BatchNormalization()(gru1)
    # gru1_out = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    #
    # gru_a = GRU(units=256,
    #             return_sequences=True,
    #             recurrent_dropout=0.5)(gru1_out)
    # x = BatchNormalization()(gru_a)
    # gru_a_out = TimeDistributed(Activation('softmax'))(x)
    #
    # dot = multiply([gru1_out, gru_a_out])
    #
    # gru2 = GRU(units=256,
    #            return_sequences=False,
    #            recurrent_dropout=0.5)(dot)
    # x = BatchNormalization()(gru2)
    # x = LeakyReLU(alpha=0.2)(x)

    # model.add(ConvLSTM2D(filters=256,
    #                     kernel_size=(3, 3),
    #                     strides=(1, 1),
    #                     padding="same",
    #                     return_sequences=True,
    #                     recurrent_dropout=0.5))
    # model.add(TimeDistributed(BatchNormalization()))
    # model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
    # model.add(TimeDistributed(Dropout(0.5)))
    #
    # model.add(ConvLSTM2D(filters=256,
    #                     kernel_size=(3, 3),
    #                     strides=(1, 1),
    #                     padding="same",
    #                     recurrent_dropout=0.5))
    # model.add(TimeDistributed(BatchNormalization()))
    # model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
    # model.add(TimeDistributed(Dropout(0.5)))
    #
    # model.add(Flatten())

    dense = Dense(units=512, activation='tanh')(c3d_out)
    x = Dropout(0.5)(dense)
    x = Dense(units=512, activation='tanh')(x)
    x = Dropout(0.5)(x)
    actions = Dense(units=len(simple_driver_set), activation='softmax')(x)
    model = Model(inputs=inputs, outputs=actions)
    # i=0
    # for layer in model.layers:
    #     print (layer, i)
    #     i = i+1
    # exit(0)

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


def stacked_classifier_model(encoder, decoder, classifier):
    input = Input(shape=(16, 112, 112, 3))
    set_trainability(encoder, True)
    z = encoder(input)
    set_trainability(decoder, True)
    future = decoder(z)
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

    img_height = video_stack.shape[3]
    img_width = video_stack.shape[2]
    # width = img_size x video_length
    width = img_width * int(VIDEO_LENGTH / 2)
    # height = img_size x batch_size
    height = img_height * BATCH_SIZE
    shape = frames.shape[1:]
    image = np.zeros((height, width, shape[2]), dtype=video_stack.dtype)
    frame_number = 0
    for i in range(BATCH_SIZE):
        for j in range(int(VIDEO_LENGTH / 2)):
            image[(i * img_height):((i + 1) * img_height), (j * img_width):((j + 1) * img_width)] = frames[frame_number]
            frame_number = frame_number + 1

    return image


def combine_images(X, y, generated_images):
    return arrange_images(X), arrange_images(y), arrange_images(generated_images)


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
                frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_CUBIC)
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
            if driver_action_nums < 2:
                driver_action_nums = 0
            elif driver_action_nums == 2:
                driver_action_nums = 1
            else:
                driver_action_nums = 2
            encoded_driver_action = to_categorical(driver_action_nums, len(simple_driver_set))
            driver_action_class.append(encoded_driver_action.T)
        else:
            driver_action_nums = driver_actions.index(action_dict['Driver'])
            if driver_action_nums < 2:
                driver_action_nums = 0
            elif driver_action_nums == 2:
                driver_action_nums = 1
            else:
                driver_action_nums = 2
            encoded_driver_action = to_categorical(driver_action_nums, len(simple_driver_set))
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
    # print (driver_action_class[0:30])
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
    videos_list = get_video_lists(frames_source=frames_source, stride=4)

    if SHUFFLE:
        # Shuffle images to aid generalization
        videos_list = np.random.permutation(videos_list)

    # Load actions from annotations
    action_labels = hkl.load(os.path.join(DATA_DIR, 'annotations_train_128.hkl'))
    driver_action_classes, ped_action_classes = get_action_classes(action_labels=action_labels)

    # Setup test
    test_frames_source = hkl.load(os.path.join(TEST_DATA_DIR, 'sources_test_128.hkl'))
    test_videos_list = get_video_lists(frames_source=test_frames_source, stride=4)
    # Load test action annotations
    test_action_labels = hkl.load(os.path.join(TEST_DATA_DIR, 'annotations_test_128.hkl'))
    test_driver_action_classes, test_ped_action_classes = get_action_classes(test_action_labels)

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

    def top_2_categorical_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=1)

        # Build stacked classifier
    if CLASSIFIER:
        classifier = pretrained_c3d()
        classifier.compile(loss="categorical_crossentropy",
                                 optimizer=OPTIM_C, metrics=['acc'])
        sclassifier = stacked_classifier_model(encoder, decoder, classifier)
        sclassifier.compile(loss=["mean_squared_error", "categorical_crossentropy"],
                            optimizer=OPTIM_C,
                            loss_weights=[1, 0],
                            metrics=['acc'])
        print (sclassifier.summary())

    run_utilities(encoder, decoder, autoencoder, classifier, ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS)

    n_videos = videos_list.shape[0]
    n_test_videos = test_videos_list.shape[0]
    NB_ITERATIONS = int(n_videos/BATCH_SIZE)
    # NB_ITERATIONS = 1
    NB_TEST_ITERATIONS = int(n_test_videos/BATCH_SIZE)
    # NB_TEST_ITERATIONS = 1

    # Setup TnsorBoard Callback
    TC = tb_callback.TensorBoard(log_dir=TF_LOG_DIR, histogram_freq=0, write_graph=False, write_images=False)
    TC_cla = tb_callback.TensorBoard(log_dir=TF_LOG_CLA_DIR, histogram_freq=0, write_graph=False, write_images=False)
    LRS = lrs_callback.LearningRateScheduler(schedule=schedule)
    # LRS.set_model(sclassifier)


    print ("Beginning Training.")
    # Begin Training
    for epoch in range(NB_EPOCHS_AUTOENCODER):
        print("\n\nEpoch ", epoch)
        loss = []
        test_loss = []

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
        for index in range(NB_TEST_ITERATIONS):
            X, y1, y2 = load_X_y(test_videos_list, index, TEST_DATA_DIR, [], [])
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
        predicted_attn = mask_gen_1.predict(X_train, verbose=0)
        a_pred = np.reshape(predicted_attn, newshape=(BATCH_SIZE, 16, 14, 14, 1))
        np.save(os.path.join(ATTN_WEIGHTS_DIR, 'attention_weights_gen1_' + str(epoch) + '.npy'), a_pred)

    # Train AAE
    if CLASSIFIER:
        # exit(0)
        print("Training Classifier...")
        for epoch in range(NB_EPOCHS_CLASS):
            print("\n\nEpoch ", epoch)
            c_loss = []
            test_c_loss = []

            # # Set learning rate every epoch
            # LRS.on_epoch_begin(epoch=epoch)
            lr = K.get_value(sclassifier.optimizer.lr)
            print("Learning rate: " + str(lr))
            print("c_loss_metrics: " + str(sclassifier.metrics_names))

            for index in range(NB_ITERATIONS):
                # Train Autoencoder
                X, y1, y2 = load_X_y(videos_list, index, DATA_DIR, driver_action_classes, [])
                X_train = X[:, 0: int(VIDEO_LENGTH / 2)]
                y_true_imgs = X[:, int(VIDEO_LENGTH / 2):]
                y1_true_classes = y1[:, int(VIDEO_LENGTH / 2):]
                y1_true_class = y1[:, int(VIDEO_LENGTH/2) + 8]
                # y2_true_classes = y2[:, int(VIDEO_LENGTH / 2):]

                c_loss.append(sclassifier.train_on_batch(X_train, [y_true_imgs, y1_true_class]))

                arrow = int(index / (NB_ITERATIONS / 30))
                stdout.write("\rIter: " + str(index) + "/" + str(NB_ITERATIONS - 1) + "  " +
                             "c_loss: " + str([ c_loss[len(c_loss) - 1][j]  for j in [0, -1]]) + "  " +
                             "\t    [" + "{0}>".format("=" * (arrow)))
                stdout.flush()

            if SAVE_GENERATED_IMAGES:
                # Save generated images to file
                predicted_images = autoencoder.predict(X_train)
                imgs, driver_pred_class = sclassifier.predict(X_train, verbose=0)
                orig_image, truth_image, pred_image = combine_images(X_train, y_true_imgs, predicted_images)
                pred_image = pred_image * 127.5 + 127.5
                orig_image = orig_image * 127.5 + 127.5
                truth_image = truth_image * 127.5 + 127.5
                font = cv2.FONT_HERSHEY_SIMPLEX
                if epoch == 0:
                    y1_orig_classes = y1[:, 0: int(VIDEO_LENGTH / 2)]
                    # y2_orig_classes = y2[:, 0: int(VIDEO_LENGTH / 2)]
                    # Add labels as text to the image
                    for k in range(BATCH_SIZE):
                        for j in range(int(VIDEO_LENGTH / 2)):
                                class_num_past_y1 = np.argmax(y1_orig_classes[k, j])
                                # class_num_past_y2 = np.argmax(y2_orig_classes[k, j])
                                class_num_futr_y1 = np.argmax(y1_true_classes[k, j])
                                # class_num_futr_y2 = np.argmax(y2_true_classes[k, j])
                                cv2.putText(orig_image, "Car: " + simple_driver_set[class_num_past_y1],
                                            (2 + j * (112), 104 + k * 112), font, 0.5, (255, 255, 255), 1,
                                            cv2.LINE_AA)
                                # cv2.putText(orig_image, "Ped: " + ped_actions[class_num_past_y2],
                                #             (2 + j * (128), 120 - 15 + k * 128), font, 0.3, (255, 255, 255), 1,
                                #             cv2.LINE_AA)
                                cv2.putText(truth_image, "Car: " + simple_driver_set[class_num_futr_y1],
                                            (2 + j * (128), 104 + k * 112), font, 0.5, (255, 255, 255), 1,
                                            cv2.LINE_AA)
                                # cv2.putText(truth_image, "Ped: " + ped_actions[class_num_futr_y2],
                                #             (2 + j * (128), 120 - 15 + k * 128), font, 0.3, (255, 255, 255), 1,
                                #             cv2.LINE_AA)
                    cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) +
                                             "_cla_orig.png"), orig_image)
                    cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) +
                                             "_cla_truth.png"), truth_image)

                # Add labels as text to the image
                for k in range(BATCH_SIZE):
                    class_num_y1 = np.argmax(driver_pred_class[k])
                    # class_num_y2 = np.argmax(ped_pred_classes[k, j])
                    cv2.putText(pred_image,  "Car: " + simple_driver_set[class_num_y1],
                                (2, 104 + k * 112), font, 0.5, (255, 255, 255), 1,
                                cv2.LINE_AA)
                    # cv2.putText(pred_image, "Ped: " +  ped_actions[class_num_y2],
                    #             (2 + j * (128), 120 - 15 + k * 128), font, 0.3, (255, 255, 255), 1,
                    #             cv2.LINE_AA)
                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_cla_pred.png"),
                            pred_image)

            # Run over test data
            print('')
            for index in range(NB_TEST_ITERATIONS):
                X, y1, y2 = load_X_y(test_videos_list, index, TEST_DATA_DIR, test_driver_action_classes, [])
                X_test = X[:, 0: int(VIDEO_LENGTH / 2)]
                y1_true_classes = y1[:, int(VIDEO_LENGTH / 2):]
                y1_true_class = y1[:, int(VIDEO_LENGTH/2) + 8]
                # y2_true_classes = y2[:, int(VIDEO_LENGTH / 2):]
                y_true_imgs = X[:, int(VIDEO_LENGTH / 2):]

                test_c_loss.append(sclassifier.test_on_batch(X_test, [y_true_imgs, y1_true_class]))

                arrow = int(index / (NB_TEST_ITERATIONS / 40))
                stdout.write("\rIter: " + str(index) + "/" + str(NB_TEST_ITERATIONS - 1) + "  " +
                             "test_c_loss: " +  str([ test_c_loss[len(test_c_loss) - 1][j]  for j in [0, -1]]))
                stdout.flush()

            # Save generated images to file
            predicted_images = autoencoder.predict(X_test)
            imgs, driver_pred_class = sclassifier.predict(X_test, verbose=0)
            orig_image, truth_image, pred_image = combine_images(X_test, y_true_imgs, predicted_images)
            pred_image = pred_image * 127.5 + 127.5
            orig_image = orig_image * 127.5 + 127.5
            truth_image = truth_image * 127.5 + 127.5
            font = cv2.FONT_HERSHEY_SIMPLEX
            if epoch == 0:
                y1_orig_classes = y1[:, 0: int(VIDEO_LENGTH / 2)]
                # y2_orig_classes = y2[:, 0: int(VIDEO_LENGTH / 2)]
                # Add labels as text to the image
                for k in range(BATCH_SIZE):
                    for j in range(int(VIDEO_LENGTH / 2)):
                        class_num_past_y1 = np.argmax(y1_orig_classes[k, j])
                        # class_num_past_y2 = np.argmax(y2_orig_classes[k, j])
                        class_num_futr_y1 = np.argmax(y1_true_classes[k, j])
                        # class_num_futr_y2 = np.argmax(y2_true_classes[k, j])
                        cv2.putText(orig_image, "Car: " + simple_driver_set[class_num_past_y1],
                                    (2 + j * (112), 104 + k * 112), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        # cv2.putText(orig_image, "Ped: " + ped_actions[class_num_past_y2],
                        #             (2 + j * (128), 120 - 15 + k * 128), font, 0.3, (255, 255, 255), 1,
                        #             cv2.LINE_AA)
                        cv2.putText(truth_image, "Car: " + simple_driver_set[class_num_futr_y1],
                                    (2 + j * (112), 104 + k * 112), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        # cv2.putText(truth_image, "Ped: " + ped_actions[class_num_futr_y2],
                        #             (2 + j * (128), 120 - 15 + k * 128), font, 0.3, (255, 255, 255), 1,
                        #             cv2.LINE_AA)
                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) +
                                         "_cla_test_orig.png"), orig_image)
                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) +
                                         "_cla_test_truth.png"), truth_image)

            # Add labels as text to the image
            for k in range(BATCH_SIZE):
                class_num_y1 = np.argmax(driver_pred_class[k])
                # class_num_y2 = np.argmax(ped_pred_classes[k, j])
                cv2.putText(pred_image, "Car: " + simple_driver_set[class_num_y1],
                            (2, 104 + k * 112), font, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
                # cv2.putText(pred_image, "Ped: " + ped_actions[class_num_y2],
                #             (2 + j * (128), 120 - 15 + k * 128), font, 0.3, (255, 255, 255), 1,
                #             cv2.LINE_AA)
            cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_cla_test_pred.png"),
                        pred_image)

            predicted_attn = mask_gen_1.predict(X_train, verbose=0)
            a_pred = np.reshape(predicted_attn, newshape=(BATCH_SIZE, 16, 14, 14, 1))
            np.save(os.path.join(ATTN_WEIGHTS_DIR, 'attention_weights_cla_gen1_' + str(epoch) + '.npy'), a_pred)

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

    # Create models
    print ("Creating models.")
    encoder = encoder_model()
    decoder = decoder_model()
    autoencoder = autoencoder_model(encoder, decoder)
    autoencoder.compile(loss='mean_squared_error', optimizer=OPTIM_A)

    classifier = pretrained_c3d()
    classifier.compile(loss="categorical_crossentropy",
                       optimizer=OPTIM_C, metrics=['accuracy'])
    sclassifier = stacked_classifier_model(encoder, decoder, classifier)
    sclassifier.compile(loss="categorical_crossentropy",
                        optimizer=OPTIM_C,
                        metrics=['accuracy'])

    run_utilities(encoder, decoder, autoencoder, classifier, ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS)

    # Build video progressions
    frames_source = hkl.load(os.path.join(TEST_DATA_DIR, 'sources_test_128.hkl'))
    videos_list = get_video_lists(frames_source=frames_source, stride=16)

    action_labels = hkl.load(os.path.join(TEST_DATA_DIR, 'annotations_test_128.hkl'))
    driver_action_classes, ped_action_classes = get_action_classes(action_labels=action_labels)
    n_videos = videos_list.shape[0]

    # Test model by making predictions
    c_loss = []
    NB_ITERATIONS = int(n_videos / BATCH_SIZE)
    for index in range(NB_ITERATIONS):
        # Test Autoencoder
        X, y1, y2 = load_X_y(videos_list, index, TEST_DATA_DIR, driver_action_classes, [])
        X_test = X[:, 0: int(VIDEO_LENGTH / 2)]
        y1_true_classes = y1[:, int(VIDEO_LENGTH / 2):]
        y1_true_class = y1[:, -1]
        y_true_imgs = X[:, int(VIDEO_LENGTH / 2):]

        c_loss.append(sclassifier.test_on_batch(X_test, y1_true_class))

        arrow = int(index / (NB_ITERATIONS / 40))
        stdout.write("\rIter: " + str(index) + "/" + str(NB_ITERATIONS - 1) + "  " +
                     "loss: " +str([ c_loss[len(c_loss) - 1][j]  for j in [0, 1]]) +
                     "\t    [" + "{0}>".format("=" * (arrow)))
        stdout.flush()

        predicted_images = autoencoder.predict(X_test)
        driver_pred_class = sclassifier.predict(X_test, verbose=0)
        orig_image, truth_image, pred_image = combine_images(X_test, y_true_imgs, predicted_images)
        pred_image = pred_image * 127.5 + 127.5
        orig_image = orig_image * 127.5 + 127.5
        truth_image = truth_image * 127.5 + 127.5

        font = cv2.FONT_HERSHEY_SIMPLEX
        y1_orig_classes = y1[:, 0: int(VIDEO_LENGTH / 2)]
        for k in range(BATCH_SIZE):
            for j in range(int(VIDEO_LENGTH / 2)):
                class_num_past_y1 = np.argmax(y1_orig_classes[k, j])
                class_num_futr_y1 = np.argmax(y1_true_classes[k, j])
                cv2.putText(orig_image, "Car: " + simple_driver_set[class_num_past_y1],
                            (2 + j * (112), 104 + k * 112), font, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
                cv2.putText(truth_image, "Car: " + simple_driver_set[class_num_futr_y1],
                            (2 + j * (112), 104 + k * 112), font, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
        cv2.imwrite(os.path.join(TEST_RESULTS_DIR, str(index) + "_" + str(index) +
                                 "_cla_orig.png"), orig_image)
        cv2.imwrite(os.path.join(TEST_RESULTS_DIR, str(index) + "_" + str(index) +
                                     "_cla_truth.png"), truth_image)

        # Add labels as text to the image
        for k in range(BATCH_SIZE):
            class_num_y1 = np.argmax(driver_pred_class[k])
            cv2.putText(pred_image, "Car: " + simple_driver_set[class_num_y1],
                        (2, 104 + k * 112), font, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
        cv2.imwrite(os.path.join(TEST_RESULTS_DIR, str(index) + "_" + str(index) + "_cla_pred.png"),
                    pred_image)

    avg_c_loss = np.mean(np.asarray(c_loss, dtype=np.float32), axis=0)
    print("\nAvg loss: " + str(avg_c_loss))


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

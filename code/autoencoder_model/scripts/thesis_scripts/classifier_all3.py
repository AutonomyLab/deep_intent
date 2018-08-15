from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hickle as hkl
import numpy as np

np.random.seed(2 ** 10)
from keras import backend as K

K.set_image_dim_ordering('tf')
import tensorflow as tf
import time
from keras import regularizers
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers.core import Activation
from keras.utils.vis_utils import plot_model
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import Conv3DTranspose
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
from sklearn.metrics import confusion_matrix
from image_utils import random_rotation
from image_utils import random_shift
from image_utils import flip_axis
from image_utils import random_brightness
from config_call3 import *
from sys import stdout
from conv_3D import encoder_model
from conv_3D import decoder_model

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
            if RANDOM_AUGMENTATION:
                frames[i] = frame.astype(np.float32)
            else:
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
    inputs = Input(shape=(int(VIDEO_LENGTH / 2), 128, 208, 3))
    z = encoder(inputs)
    future = decoder(z)

    model = Model(inputs=inputs, outputs=future)

    return model


def stacked_classifier_model(encoder, decoder, classifier):
    input = Input(shape=(16, 128, 208, 3))
    set_trainability(encoder, FINETUNE_ENCODER)
    z = encoder(input)
    set_trainability(decoder, FINETUNE_DECODER)
    future = decoder(z)
    set_trainability(classifier, FINETUNE_CLASSIFIER)
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

    img_height = video_stack.shape[2]
    img_width = video_stack.shape[3]
    width = img_width * video_stack.shape[1]
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
        print("Encoder:")
        print(encoder.summary())
        print("Decoder:")
        print(decoder.summary())
        if CLASSIFIER:
            print("Classifier:")
            print(classifier.summary())

    # Save model to file
    if SAVE_MODEL:
        print("Saving models to file.")
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
        print("Pre-loading encoder with weights.")
        load_weights(ENC_WEIGHTS, encoder)
    if DEC_WEIGHTS != "None":
        print("Pre-loading decoder with weights.")
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
        for i in range(VIDEO_LENGTH):
            video[i] = random_rotation(video[i], (i * theta) / VIDEO_LENGTH, row_axis=0,
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
            video = np.take(frames, videos_list[(index * BATCH_SIZE + i)], axis=0)
            if RANDOM_AUGMENTATION:
                video = random_augmentation(video)
            X.append(video)
            if (len(ped_action_cats) != 0):
                y.append(np.take(ped_action_cats, videos_list[(index * BATCH_SIZE + i)], axis=0))

        X = np.asarray(X)
        y = np.asarray(y)
        return X, y
    else:
        print("RAM usage flag not set. Are you sure you want to do this?")
        exit(0)


def load_X_y(videos_list, index, data_dir, ped_action_cats, batch_size=BATCH_SIZE):
    X = np.zeros((batch_size, VIDEO_LENGTH,) + IMG_SIZE)
    y = []
    for i in range(batch_size):
        y_per_vid = []
        for j in range(VIDEO_LENGTH):
            frame_number = (videos_list[(index * batch_size + i), j])
            filename = "frame_" + str(frame_number) + ".png"
            im_file = os.path.join(data_dir, filename)
            try:
                frame = cv2.imread(im_file, cv2.IMREAD_COLOR)
                X[i, j] = (frame.astype(np.float32) - 127.5) / 127.5
            except AttributeError as e:
                print(im_file)
                print(e)
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
        for key, value in action_dict.items():
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
                print("Unknown action in labels. Exiting.")
                print(action)
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
            videos_list.append(range(start_frame_index, end_frame_index, frame_skip + 1))
            start_frame_index = start_frame_index + stride
            end_frame_index = end_frame_index + stride
        else:
            start_frame_index = end_frame_index - 1
            end_frame_index = start_frame_index + (frame_skip + 1) * VIDEO_LENGTH - frame_skip

    videos_list = np.asarray(videos_list, dtype=np.int32)

    return np.asarray(videos_list)


def get_sklearn_metrics(y_true, y_pred, avg=None, pos_label=1):
    return precision_recall_fscore_support(y_true, np.round(y_pred), average=avg, pos_label=pos_label)


def get_classification_report(y_true, y_pred):
    return classification_report(y_true, np.round(y_pred), target_names=['crossing', 'not crossing'])


def train(BATCH_SIZE, ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS):
    print("Loading data definitions.")

    frames_source = hkl.load(os.path.join(DATA_DIR, 'sources_train_208.hkl'))
    # videos_list_1 = get_video_lists(frames_source=frames_source, stride=8, frame_skip=0)
    videos_list = get_video_lists(frames_source=frames_source, stride=8, frame_skip=0)
    # videos_list_2 = get_video_lists(frames_source=frames_source, stride=8, frame_skip=1)
    # videos_list = np.concatenate((videos_list_1, videos_list_2), axis=0)

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
    val_action_labels = hkl.load(os.path.join(VAL_DATA_DIR, 'annotations_val_208.hkl'))
    val_ped_action_classes, val_ped_class_count = get_action_classes(val_action_labels)
    print("Val Stats: " + str(val_ped_class_count))

    # Build the Spatio-temporal Autoencoder
    print("Creating models.")
    encoder = encoder_model()
    decoder = decoder_model()

    # Build stacked classifier
    classifier = ensemble_c3d()
    run_utilities(encoder, decoder, classifier, ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS)
    sclassifier = stacked_classifier_model(encoder, decoder, classifier)
    sclassifier.compile(loss=["binary_crossentropy"],
                        optimizer=OPTIM_C,
                        metrics=['accuracy'])
    print(sclassifier.summary())

    n_videos = videos_list.shape[0]
    n_val_videos = val_videos_list.shape[0]
    NB_ITERATIONS = int(n_videos / BATCH_SIZE)
    # NB_ITERATIONS = 1
    NB_VAL_ITERATIONS = int(n_val_videos / BATCH_SIZE)
    # NB_VAL_ITERATIONS = 1

    # Setup TensorBoard Callback
    TC_cla = tb_callback.TensorBoard(log_dir=TF_LOG_CLA_DIR, histogram_freq=0, write_graph=False, write_images=False)
    LRS_cla = lrs_callback.LearningRateScheduler(schedule=cla_schedule)
    LRS_cla.set_model(sclassifier)

    print("Beginning Training.")
    # Begin Training

    # Train Classifier
    print("Training Classifier...")
    for epoch in range(1, NB_EPOCHS_CLASS + 1):
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

            c_loss.append(sclassifier.train_on_batch(X_train, y_true_class))

            y_train_true.extend(y_true_class)
            y_train_pred.extend(sclassifier.predict(X_train, verbose=0))

            arrow = int(index / (NB_ITERATIONS / 30))
            stdout.write("\rIter: " + str(index) + "/" + str(NB_ITERATIONS - 1) + "  " +
                         "c_loss: " + str([c_loss[len(c_loss) - 1][j] for j in [0, 1]]) + "  " +
                         "\t    [" + "{0}>".format("=" * (arrow)))
            stdout.flush()

        if SAVE_GENERATED_IMAGES:
            # Save generated images to file
            z = encoder.predict(X_train)
            predicted_images = decoder.predict(z)
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
                    if y_orig_classes[k, j] > 0.5:
                        label_orig = "crossing"
                    else:
                        label_orig = "not crossing"

                    if y_true_classes[k][0] > 0.5:
                        label_true = "crossing"
                    else:
                        label_true = "not crossing"

                    if ped_pred_class[k][0] > 0.5:
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

        # Run over validation data
        print('')
        y_val_pred = []
        y_val_true = []
        for index in range(NB_VAL_ITERATIONS):
            X, y = load_X_y(val_videos_list, index, VAL_DATA_DIR, val_ped_action_classes)
            X_val = np.flip(X[:, 0: int(VIDEO_LENGTH / 2)], axis=1)
            y_true_class = y[:, CLASS_TARGET_INDEX]
            y_true_imgs = X[:, int(VIDEO_LENGTH / 2):]

            val_c_loss.append(sclassifier.test_on_batch(X_val, y_true_class))
            y_val_true.extend(y_true_class)
            y_val_pred.extend(sclassifier.predict(X_val, verbose=0))

            arrow = int(index / (NB_VAL_ITERATIONS / 40))
            stdout.write("\rIter: " + str(index) + "/" + str(NB_VAL_ITERATIONS - 1) + "  " +
                         "val_c_loss: " + str([val_c_loss[len(val_c_loss) - 1][j] for j in [0, 1]]))
            stdout.flush()

        # Save generated images to file
        z = encoder.predict(X_val)
        val_predicted_images = decoder.predict(z)
        val_ped_pred_class = sclassifier.predict(X_val, verbose=0)
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
                                     "_cla_val_orig.png"), orig_image)
            cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) +
                                     "_cla_val_truth.png"), truth_image)

        # Add labels as text to the image
        for k in range(BATCH_SIZE):
            # class_num_y = np.argmax(val_ped_pred_class[k])
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
                                                                                avg='binary',
                                                                                pos_label=1)
        val_prec, val_rec, val_fbeta, val_support = get_sklearn_metrics(np.asarray(y_val_true),
                                                                        np.asarray(y_val_pred),
                                                                        avg='binary',
                                                                        pos_label=1)

        print("\nTrain Prec: %.2f, Recall: %.2f, Fbeta: %.2f" % (train_prec, train_rec, train_fbeta))
        print("Val Prec: %.2f, Recall: %.2f, Fbeta: %.2f" % (val_prec, val_rec, val_fbeta))
        loss_values = np.asarray(avg_c_loss.tolist() + [train_prec.tolist()] +
                                 [train_rec.tolist()] +
                                 avg_val_c_loss.tolist() + [val_prec.tolist()] +
                                 [val_rec.tolist()], dtype=np.float32)
        # loss_values = np.asarray(avg_c_loss.tolist() + train_prec.tolist() +
        #                          train_rec.tolist() +
        #                          avg_val_c_loss.tolist() + val_prec.tolist() +
        #                          val_rec.tolist(), dtype=np.float32)

        precs = ['prec_' + action for action in simple_ped_set]
        recs = ['rec_' + action for action in simple_ped_set]
        c_loss_keys = ['c_' + metric for metric in sclassifier.metrics_names + precs + recs]
        val_c_loss_keys = ['c_val_' + metric for metric in sclassifier.metrics_names + precs + recs]

        loss_keys = c_loss_keys + val_c_loss_keys
        logs = dict(zip(loss_keys, loss_values))

        TC_cla.on_epoch_end(epoch, logs)

        # Log the losses
        with open(os.path.join(LOG_DIR, 'losses_cla.json'), 'a') as log_file:
            log_file.write("{\"epoch\":%d, %s;\n" % (epoch, logs))

        print("\nAvg c_loss: " + str(avg_c_loss) +
              " Avg val_c_loss: " + str(avg_val_c_loss))

        # Save model weights per epoch to file
        if FINETUNE_ENCODER:
            encoder.save_weights(os.path.join(CHECKPOINT_DIR, 'encoder_cla_epoch_' + str(epoch) + '.h5'), True)
        if FINETUNE_DECODER:
            decoder.save_weights(os.path.join(CHECKPOINT_DIR, 'decoder_cla_epoch_' + str(epoch) + '.h5'), True)
        classifier.save_weights(os.path.join(CHECKPOINT_DIR, 'classifier_cla_epoch_' + str(epoch) + '.h5'),
                                True)
    print(get_classification_report(np.asarray(y_train_true), np.asarray(y_train_pred)))
    print(get_classification_report(np.asarray(y_val_true), np.asarray(y_val_pred)))


def test(ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS):
    if not os.path.exists(TEST_RESULTS_DIR + '/pred/'):
        os.mkdir(TEST_RESULTS_DIR + '/pred/')
    if not os.path.exists(TEST_RESULTS_DIR + '/truth/'):
        os.mkdir(TEST_RESULTS_DIR + '/truth/')

    # Setup test
    test_frames_source = hkl.load(os.path.join(TEST_DATA_DIR, 'sources_test_208.hkl'))
    test_videos_list = get_video_lists(frames_source=test_frames_source, stride=16, frame_skip=0)
    # Load test action annotations
    test_action_labels = hkl.load(os.path.join(TEST_DATA_DIR, 'annotations_test_208.hkl'))
    test_ped_action_classes, test_ped_class_count = get_action_classes(test_action_labels)
    print("Test Stats: " + str(test_ped_class_count))

    # Build the Spatio-temporal Autoencoder
    print("Creating models.")

    # Build stacked classifier
    encoder = encoder_model()
    decoder = decoder_model()

    # Build stacked classifier
    classifier = ensemble_c3d()
    run_utilities(encoder, decoder, classifier, ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS)

    sclassifier = stacked_classifier_model(encoder, decoder, classifier)
    sclassifier.compile(loss=["binary_crossentropy"],
                        optimizer=OPTIM_C,
                        metrics=['accuracy'])
    print(sclassifier.summary())

    n_test_videos = test_videos_list.shape[0]

    NB_TEST_ITERATIONS = int(n_test_videos / TEST_BATCH_SIZE)
    # NB_TEST_ITERATIONS = 5

    if CLASSIFIER:
        print("Testing Classifier...")
        # Run over test data
        print('')
        y_test_pred = []
        y_test_true = []
        test_c_loss = []
        iter_loadtime = []
        iter_starttime = []
        iter_endtime = []
        for index in range(NB_TEST_ITERATIONS):
            iter_loadtime.append(time.time())
            X, y = load_X_y(test_videos_list, index, TEST_DATA_DIR, test_ped_action_classes,
                            batch_size=TEST_BATCH_SIZE)
            X_test = np.flip(X[:, 0: int(VIDEO_LENGTH / 2)], axis=1)
            y_true_class = y[:, CLASS_TARGET_INDEX]
            y_true_imgs = X[:, int(VIDEO_LENGTH / 2):]

            iter_starttime.append(time.time())
            test_ped_pred_class = sclassifier.predict(X_test, verbose=0)
            iter_endtime.append(time.time())

            test_c_loss.append(sclassifier.test_on_batch(X_test, y_true_class))
            y_test_true.extend(y_true_class)
            y_test_pred.extend(sclassifier.predict(X_test, verbose=0))

            arrow = int(index / (NB_TEST_ITERATIONS / 40))
            stdout.write("\rIter: " + str(index) + "/" + str(NB_TEST_ITERATIONS - 1) + "  " +
                         "test_c_loss: " + str([test_c_loss[len(test_c_loss) - 1][j] for j in [0, 1]]))
            stdout.flush()

            # Save generated images to file
            z = encoder.predict(X_test)
            test_predicted_images = decoder.predict(z)
            test_ped_pred_class = sclassifier.predict(X_test, verbose=0)
            pred_seq = arrange_images(np.concatenate((X_test, test_predicted_images), axis=1))
            pred_seq = pred_seq * 127.5 + 127.5

            truth_image = arrange_images(y_true_imgs)
            truth_image = truth_image * 127.5 + 127.5

            font = cv2.FONT_HERSHEY_SIMPLEX
            y_orig_classes = y[:, 0: int(VIDEO_LENGTH / 2)]
            y_true_classes = y[:, int(VIDEO_LENGTH / 2):]

            # Add labels as text to the image
            for k in range(TEST_BATCH_SIZE):
                for j in range(int(VIDEO_LENGTH / 2)):
                    if y_orig_classes[k, j] > 0.5:
                        label_orig = "crossing"
                    else:
                        label_orig = "not crossing"

                    if y_true_classes[k][0] > 0.5:
                        label_true = "crossing"
                    else:
                        label_true = "not crossing"

                    if test_ped_pred_class[k][0] > 0.5:
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

            cv2.imwrite(os.path.join(TEST_RESULTS_DIR + '/pred/', str(index) + "_cla_test_pred.png"), pred_seq)
            cv2.imwrite(os.path.join(TEST_RESULTS_DIR + '/truth/', str(index) + "_cla_test_truth.png"), truth_image)

        # then after each epoch
        avg_test_c_loss = np.mean(np.asarray(test_c_loss, dtype=np.float32), axis=0)

        test_prec, test_rec, test_fbeta, test_support = get_sklearn_metrics(np.asarray(y_test_true),
                                                                            np.asarray(y_test_pred),
                                                                            avg='binary',
                                                                            pos_label=1)
        print("\nAvg test_c_loss: " + str(avg_test_c_loss))
        print("Test Prec: %.4f, Recall: %.4f, Fbeta: %.4f" % (test_prec, test_rec, test_fbeta))

        print("Classification Report")
        print(get_classification_report(np.asarray(y_test_true), np.asarray(y_test_pred)))

        print("Confusion matrix")
        tn, fp, fn, tp = confusion_matrix(y_test_true, np.round(y_test_pred)).ravel()
        print("TN: %.2f, FP: %.2f, FN: %.2f, TP: %.2f" % (tn, fp, fn, tp))

        print("Mean time taken to make " + str(NB_TEST_ITERATIONS) + " predictions: %f"
              % (np.mean(np.asarray(iter_endtime) - np.asarray(iter_starttime))))
        print("Standard Deviation %f"
              % (np.std(np.asarray(iter_endtime) - np.asarray(iter_starttime))))

        print("Mean time taken to make load and process" + str(NB_TEST_ITERATIONS) + " predictions: %f"
              % (np.mean(np.asarray(iter_endtime) - np.asarray(iter_loadtime))))
        print("Standard Deviation %f"
              % (np.std(np.asarray(iter_endtime) - np.asarray(iter_loadtime))))


def test_mtcp(ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS):
    if not os.path.exists(TEST_RESULTS_DIR + '/mtcp-pred/'):
        os.mkdir(TEST_RESULTS_DIR + '/mtcp-pred/')
    if not os.path.exists(TEST_RESULTS_DIR + '/mtcp-truth/'):
        os.mkdir(TEST_RESULTS_DIR + '/mtcp-truth/')

    # Setup test
    test_frames_source = hkl.load(os.path.join(TEST_DATA_DIR, 'sources_test_208.hkl'))
    test_videos_list = get_video_lists(frames_source=test_frames_source, stride=1, frame_skip=0)
    # Load test action annotations
    test_action_labels = hkl.load(os.path.join(TEST_DATA_DIR, 'annotations_test_208.hkl'))
    test_ped_action_classes, test_ped_class_count = get_action_classes(test_action_labels)
    print("Test Stats: " + str(test_ped_class_count))

    # Build the Spatio-temporal Autoencoder
    print("Creating models.")

    # Build stacked classifier
    encoder = encoder_model()
    decoder = decoder_model()

    # Build stacked classifier
    classifier = ensemble_c3d()
    run_utilities(encoder, decoder, classifier, ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS)

    sclassifier = stacked_classifier_model(encoder, decoder, classifier)
    sclassifier.compile(loss=["binary_crossentropy"],
                        optimizer=OPTIM_C,
                        metrics=['accuracy'])
    print(sclassifier.summary())

    n_test_videos = test_videos_list.shape[0]

    NB_TEST_ITERATIONS = int(n_test_videos / TEST_BATCH_SIZE)
    print ("Number of iterations %d" %NB_TEST_ITERATIONS)
    # NB_TEST_ITERATIONS = 5

    if CLASSIFIER:
        print("Testing Classifier...")
        # Run over test data
        print('')
        # Time to correct prediction
        tcp_list = []
        y_test_pred = []
        y_test_true = []
        test_c_loss = []
        index = 0
        tcp = 1
        while index < NB_TEST_ITERATIONS:
            stdout.write("\rIter: " + str(index) + "/" + str(NB_TEST_ITERATIONS - 1))
            stdout.flush()
            X, y = load_X_y(test_videos_list, index, TEST_DATA_DIR, test_ped_action_classes,
                            batch_size=TEST_BATCH_SIZE)
            X_test = np.flip(X[:, 0: int(VIDEO_LENGTH / 2)], axis=1)
            y_true_imgs = X[:, int(VIDEO_LENGTH / 2):]

            y_past_class = y[:, 0]
            y_end_class = y[:,-1]

            if y_end_class[0] == y_past_class[0]:
                index = index + 1
                continue
            else:
                for fnum in range (int(VIDEO_LENGTH/2)):

                    X, y = load_X_y(test_videos_list, index, TEST_DATA_DIR, test_ped_action_classes,
                                    batch_size=TEST_BATCH_SIZE)
                    X_test = np.flip(X[:, 0: int(VIDEO_LENGTH / 2)], axis=1)
                    y_true_imgs = X[:, int(VIDEO_LENGTH / 2):]
                    y_true_class = y[:, VIDEO_LENGTH - fnum - 1]
                    if y[:, 0] == y_true_class[0]:
                        break

                    y_pred_class = sclassifier.predict(X_test, verbose=0)
                    y_test_pred.extend(sclassifier.predict(X_test, verbose=0))
                    test_c_loss.append(sclassifier.test_on_batch(X_test, y_true_class))
                    y_test_true.extend(y_true_class)

                    # Save generated images to file
                    z = encoder.predict(X_test)
                    test_predicted_images = decoder.predict(z)
                    test_ped_pred_class = sclassifier.predict(X_test, verbose=0)
                    pred_seq = arrange_images(np.concatenate((X_test, test_predicted_images), axis=1))
                    pred_seq = pred_seq * 127.5 + 127.5

                    truth_image = arrange_images(y_true_imgs)
                    truth_image = truth_image * 127.5 + 127.5

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    y_orig_classes = y[:, 0: int(VIDEO_LENGTH / 2)]
                    y_true_classes = y[:, int(VIDEO_LENGTH / 2):]

                    # Add labels as text to the image
                    for k in range(TEST_BATCH_SIZE):
                        for j in range(int(VIDEO_LENGTH / 2)):
                            if y_orig_classes[k, j] > 0.5:
                                label_orig = "crossing"
                            else:
                                label_orig = "not crossing"

                            if y_true_classes[k][j] > 0.5:
                                label_true = "crossing"
                            else:
                                label_true = "not crossing"

                            if test_ped_pred_class[k][0] > 0.5:
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

                    cv2.imwrite(os.path.join(TEST_RESULTS_DIR + '/mtcp-pred//', str(index) + "_cla_test_pred.png"),
                                pred_seq)
                    cv2.imwrite(os.path.join(TEST_RESULTS_DIR + '/mtcp-truth/', str(index) + "_cla_test_truth.png"),
                                truth_image)

                    if y_true_class[0] != np.round(y_pred_class[0]):
                        index = index + 1
                        continue
                    else:
                        tcp_list.append(fnum + 1)
                        index = index + int(VIDEO_LENGTH / 2)
                        # Break from the for loop
                        break


        # then after each epoch
        avg_test_c_loss = np.mean(np.asarray(test_c_loss, dtype=np.float32), axis=0)

        test_prec, test_rec, test_fbeta, test_support = get_sklearn_metrics(np.asarray(y_test_true),
                                                                            np.asarray(y_test_pred),
                                                                            avg='binary',
                                                                            pos_label=1)
        print("\nAvg test_c_loss: " + str(avg_test_c_loss))
        print("Mean time to change prediction: " + str(np.mean(np.asarray(tcp_list))))
        print("Standard Deviation " + str(np.std(np.asarray(tcp_list))))
        print ("Number of correct predictions " + str(len(tcp_list)))
        print("Test Prec: %.4f, Recall: %.4f, Fbeta: %.4f" % (test_prec, test_rec, test_fbeta))

        print("Classification Report")
        print(get_classification_report(np.asarray(y_test_true), np.asarray(y_test_pred)))

        print("Confusion matrix")
        tn, fp, fn, tp = confusion_matrix(y_test_true, np.round(y_test_pred)).ravel()
        print("TN: %.2f, FP: %.2f, FN: %.2f, TP: %.2f" % (tn, fp, fn, tp))


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

    if args.mode == "test-mtcp":
        test_mtcp(ENC_WEIGHTS=args.enc_weights,
                  DEC_WEIGHTS=args.dec_weights,
                  CLA_WEIGHTS=args.cla_weights)

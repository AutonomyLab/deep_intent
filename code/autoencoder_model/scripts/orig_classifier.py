from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hickle as hkl
import numpy as np

np.random.seed(2 ** 10)
import tensorflow as tf
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
from sklearn.metrics import precision_recall_fscore_support as score
from config_oc import *
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


def c3d_scratch():
    model = Sequential()
    model.add(Conv3D(filters=64,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='conv1',
                     input_shape=(VIDEO_LENGTH, 128, 128, 3)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(filters=128,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='conv2'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(filters=256,
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
    model.add(Dense(1024, activation='relu', name='fc6'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', name='fc7'))
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


def load_X_y_RAM(videos_list, index, frames, driver_action_cats, ped_action_cats):
    # X = np.zeros((BATCH_SIZE, VIDEO_LENGTH,) + IMG_SIZE)
    X = []
    y1 = []
    y2 = []
    for i in range(BATCH_SIZE):
        start_index = videos_list[(index*BATCH_SIZE + i), 0]
        end_index = videos_list[(index*BATCH_SIZE + i), -1]
        X.append(frames[start_index:end_index+1])
        if (len(driver_action_cats) != 0):
            y1.append(driver_action_cats[start_index:end_index+1])
        if (len(ped_action_cats) != 0):
            y2.append(ped_action_cats[start_index:end_index+1])

    X = np.asarray(X)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    return X, y1, y2


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
                # frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_LANCZOS4)
                # frame = cv2.medianBlur(frame, 5)
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # convert it to hsv
                # frame[:, :, 2] -= 2
                # frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
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


def map_to_simple(ped_action):
    if (ped_action == 0):
        return 0
    elif (ped_action == 1):
        return 1
    elif (ped_action == 2):
        return 1
    elif (ped_action == 3):
        return 2
    elif (ped_action == 4):
        return 2
    elif (ped_action == 5):
        return 3
    elif (ped_action == 6):
        return 3
    elif (ped_action == 7):
        return 3
    elif (ped_action == 8):
        return 4
    elif (ped_action == 9):
        return 5
    elif (ped_action == 10):
        return 3
    elif (ped_action == 11):
        return 3
    elif (ped_action == 12):
        return 6
    else:
        print (ped_action)


def get_action_classes(action_labels):
    # Load labesl into categorical 1-hot vectors
    print("Loading annotations.")
    count = [0] * len(simple_ped_set)
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
        encoded_ped_action = np.zeros(shape=(len(simple_ped_set)), dtype=np.float32)
        simple_ped_action_per_frame = []
        for action in ped_action_per_frame:
            # Get ped action number and map it to simple set
            ped_action = ped_actions.index(action)
            ped_action = map_to_simple(ped_action)
            simple_ped_action_per_frame.append(ped_action)


        simple_ped_action_per_frame = set(simple_ped_action_per_frame)
        for action in simple_ped_action_per_frame:
            count[action] = count[action] + 1
            # Add all unique categorical one-hot vectors
            encoded_ped_action = encoded_ped_action + to_categorical(action, len(simple_ped_set))

        if (sum(encoded_ped_action) == 0):
            print (simple_ped_action_per_frame)
            print (a_clean)
        ped_action_class.append(encoded_ped_action.T)

    driver_action_class = np.asarray(driver_action_class)
    driver_action_class = np.reshape(driver_action_class, newshape=(driver_action_class.shape[0:2]))

    ped_action_class = np.asarray(ped_action_class)
    ped_action_class = np.reshape(ped_action_class, newshape=(ped_action_class.shape[0:2]))

    return driver_action_class, ped_action_class, count


def load_to_RAM(frames_source):
    frames = np.zeros(shape=((len(frames_source),) + IMG_SIZE))

    j = 1
    for i in range(1, len(frames_source)):
        filename = "frame_" + str(j) + ".png"
        im_file = os.path.join(DATA_DIR, filename)
        try:
            frame = cv2.imread(im_file, cv2.IMREAD_COLOR)
            # frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_LANCZOS4)
            # frame = cv2.medianBlur(frame, 5)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # convert it to hsv
            # frame[:, :, 2] -= 2
            # frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            frames[i] = (frame.astype(np.float32) - 127.5) / 127.5
            j = j + 1
        except AttributeError as e:
            print(im_file)
            print(e)

    return frames


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


def subsample_videos(videos_list, ped_action_labels):
    print (videos_list.shape)
    AP_MAX = 5
    LK_MAX = 4
    CR_MAX = 12
    UK_MAX = 4

    ap_count = 0
    lk_count = 0
    cr_count = 0
    uk_count = 0

    r_indices = []

    count = [0] * len(simple_ped_set)
    for i in range(len(videos_list)):
        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 0):
            count[0] = count[0] + 1

        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 1):
            count[1] = count[1] + 1

        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 2):
            count[2] = count[2] + 1

        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 3):
            count[3] = count[3] + 1

        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 4):
            count[4] = count[4] + 1

        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 5):
            count[5] = count[5] + 1

        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 6):
            count[6] = count[6] + 1


    print('Before subsampling')
    print(str(count))

    for i in range(len(videos_list)):
        # Approaching count
        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 1):
            ap_count = ap_count + 1
            if (ap_count < AP_MAX):
                r_indices.append(i)
            else:
                ap_count = 0

        # Looking count
        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 2):
            lk_count = lk_count + 1
            if (lk_count < LK_MAX):
                r_indices.append(i)
            else:
                lk_count = 0

        # Crossing count
        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 3):
            cr_count = cr_count + 1
            if (cr_count < CR_MAX):
                r_indices.append(i)
            else:
                cr_count = 0

        # Unknown count
        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 6):
            uk_count = uk_count + 1
            if (uk_count < UK_MAX):
                r_indices.append(i)
            else:
                uk_count = 0


    for i in sorted(r_indices, reverse=True):
        videos_list = np.delete(videos_list, i, axis=0)

    count = [0] * len(simple_ped_set)
    for i in range(len(videos_list)):
        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 0):
            count[0] = count[0] + 1

        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 1):
            count[1] = count[1] + 1

        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 2):
            count[2] = count[2] + 1

        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 3):
            count[3] = count[3] + 1

        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 4):
            count[4] = count[4] + 1

        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 5):
            count[5] = count[5] + 1

        if (list(ped_action_labels[videos_list[i, CLASS_TARGET_INDEX]]).index(1) == 6):
            count[6] = count[6] + 1

    print ('After subsampling')
    print (str(count))

    return videos_list


def train(BATCH_SIZE, ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS):
    print("Loading data definitions.")
    frames_source = hkl.load(os.path.join(DATA_DIR, 'sources_train_208.hkl'))
    videos_list = get_video_lists(frames_source=frames_source, stride=1)

    # Load actions from annotations
    action_labels = hkl.load(os.path.join(DATA_DIR, 'annotations_train_208.hkl'))
    driver_action_classes, ped_action_classes, ped_class_count = get_action_classes(action_labels=action_labels)
    print("Training Stats: " + str(ped_class_count))

    if RAM_DECIMATE:
        frames = load_to_RAM(frames_source=frames_source)

    if SHUFFLE:
        # Shuffle images to aid generalization
        videos_list = np.random.permutation(videos_list)

    # Setup test
    test_frames_source = hkl.load(os.path.join(TEST_DATA_DIR, 'sources_test_208.hkl'))
    test_videos_list = get_video_lists(frames_source=test_frames_source, stride=8)
    # Load test action annotations
    test_action_labels = hkl.load(os.path.join(TEST_DATA_DIR, 'annotations_test_208.hkl'))
    test_driver_action_classes, test_ped_action_classes, test_ped_class_count = get_action_classes(test_action_labels)
    print ("Test Stats: " + str(test_ped_class_count))

    videos_list = subsample_videos(videos_list=videos_list, ped_action_labels=ped_action_classes)
    # Build the Spatio-temporal Autoencoder
    print ("Creating models.")

    def top_2_categorical_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=2)

        # Build stacked classifier
    if CLASSIFIER:
        classifier = pretrained_c3d()
        # classifier = clstm_classifier()
        # classifier = c3d_scratch()
        classifier.compile(loss="binary_crossentropy",
                           optimizer=OPTIM_C,
                           metrics=[metric_precision, metric_recall, metric_mpca, 'accuracy'])

    run_utilities(classifier, CLA_WEIGHTS)

    n_videos = videos_list.shape[0]
    n_test_videos = test_videos_list.shape[0]
    NB_ITERATIONS = int(n_videos/BATCH_SIZE)
    # NB_ITERATIONS = 1
    NB_TEST_ITERATIONS = int(n_test_videos/BATCH_SIZE)
    # NB_TEST_ITERATIONS = 1

    # Setup TnsorBoard Callback
    TC_cla = tb_callback.TensorBoard(log_dir=TF_LOG_CLA_DIR, histogram_freq=0, write_graph=False, write_images=False)
    LRS = lrs_callback.LearningRateScheduler(schedule=schedule)
    LRS.set_model(classifier)


    print ("Beginning Training.")
    # Begin Training
    if CLASSIFIER:
        # exit(0)
        print("Training Classifier...")
        for epoch in range(NB_EPOCHS_CLASS):
            print("\n\nEpoch ", epoch)
            c_loss = []
            test_c_loss = []

            # # Set learning rate every epoch
            LRS.on_epoch_begin(epoch=epoch)
            lr = K.get_value(classifier.optimizer.lr)
            print("Learning rate: " + str(lr))
            print("c_loss_metrics: " + str(classifier.metrics_names))

            for index in range(NB_ITERATIONS):
                # Train Autoencoder
                if RAM_DECIMATE:
                    X, y1, y2 = load_X_y_RAM(videos_list, index, frames, [], ped_action_classes)
                else:
                    X, y1, y2 = load_X_y(videos_list, index, DATA_DIR, [], ped_action_classes)

                X_train = X
                y2_true_class = y2[:, CLASS_TARGET_INDEX]

                c_loss.append(classifier.train_on_batch(X_train, y2_true_class))

                arrow = int(index / (NB_ITERATIONS / 30))
                stdout.write("\rIter: " + str(index) + "/" + str(NB_ITERATIONS - 1) + "  " +
                             "c_loss: " + str([ c_loss[len(c_loss) - 1][j]  for j in [0, 1, 2, 3, 4]]) + "  " +
                             "\t    [" + "{0}>".format("=" * (arrow)))
                stdout.flush()

            if SAVE_GENERATED_IMAGES:
                # Save generated images to file
                ped_pred_class = classifier.predict(X_train, verbose=0)
                orig_image = arrange_images(X_train)
                orig_image = orig_image * 127.5 + 127.5
                pred_image = np.copy(orig_image)
                font = cv2.FONT_HERSHEY_SIMPLEX
                if epoch == 0:
                    y2_orig_classes = y2
                    # Add labels as text to the image
                    for k in range(BATCH_SIZE):
                        for j in range(VIDEO_LENGTH):
                                class_num_past_y2 = np.argmax(y2_orig_classes[k, j])
                                cv2.putText(orig_image, simple_ped_set[class_num_past_y2],
                                            (2 + j * (208), 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                                            cv2.LINE_AA)
                    cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) +
                                             "_cla_orig.png"), orig_image)

                # Add labels as text to the image
                for k in range(BATCH_SIZE):
                    class_num_y2 = np.argmax(ped_pred_class[k])
                    cv2.putText(pred_image,  simple_ped_set[class_num_y2],
                                (2, 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                                cv2.LINE_AA)
                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_cla_pred.png"),
                            pred_image)

            # Run over test data
            print('')
            for index in range(NB_TEST_ITERATIONS):
                X, y1, y2 = load_X_y(test_videos_list, index, TEST_DATA_DIR, [], test_ped_action_classes)
                X_test = X
                y2_true_class = y2[:, CLASS_TARGET_INDEX]

                test_c_loss.append(classifier.test_on_batch(X_test, y2_true_class))

                arrow = int(index / (NB_TEST_ITERATIONS / 40))
                stdout.write("\rIter: " + str(index) + "/" + str(NB_TEST_ITERATIONS - 1) + "  " +
                             "test_c_loss: " +  str([ test_c_loss[len(test_c_loss) - 1][j]  for j in [0, 1, 2, 3, 4]]))
                stdout.flush()

            # Save generated images to file
            ped_pred_class = classifier.predict(X_test, verbose=0)
            orig_image = arrange_images(X_test)
            orig_image = orig_image * 127.5 + 127.5
            pred_image = np.copy(orig_image)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if epoch == 0:
                y2_orig_classes = y2
                # Add labels as text to the image
                for k in range(BATCH_SIZE):
                    for j in range(VIDEO_LENGTH):
                        class_num_past_y2 = np.argmax(y2_orig_classes[k, j])
                        cv2.putText(orig_image, simple_ped_set[class_num_past_y2],
                                    (2 + j * (112), 104 + k * 112), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) +
                                         "_cla_test_orig.png"), orig_image)

            # Add labels as text to the image
            for k in range(BATCH_SIZE):
                class_num_y2 = np.argmax(ped_pred_class[k])
                cv2.putText(pred_image, simple_ped_set[class_num_y2],
                            (2, 104 + k * 112), font, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
            cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_cla_test_pred.png"),
                        pred_image)

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

            classifier.save_weights(os.path.join(CHECKPOINT_DIR, 'classifier_cla_epoch_' + str(epoch) + '.h5'),
                                    True)


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

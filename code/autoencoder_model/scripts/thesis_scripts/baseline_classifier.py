from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib.pyplot import axes

import hickle as hkl
import numpy as np

np.random.seed(2 ** 10)
import tensorflow as tf
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras import regularizers
from keras.layers import Dropout
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import concatenate
from keras.layers.core import Dense
from keras.layers.core import Lambda
from keras.layers.core import Flatten
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.layers import Input
from keras.models import Model
from keras.models import model_from_json
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from image_utils import random_rotation
from image_utils import random_shift
from image_utils import flip_axis
from image_utils import random_brightness
from config_basec import *
from sys import stdout

import tb_callback
import lrs_callback
import argparse
import time
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
    inputs = Input(shape=(16, 128, 112, 3))
    resized = TimeDistributed(Lambda(lambda image: tf.image.resize_images(image, (112, 112))))(inputs)
    c3d_out = c3d(resized)
    model = Model(inputs=inputs, outputs=c3d_out)

    # i = 0
    # for layer in model.layers:
    #     print(layer, i)
    #     i = i + 1

    print (c3d.summary())

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
    c3d_A.compile(loss="binary_crossentropy",
                optimizer=OPTIM_C)
    c3d_B.compile(loss="binary_crossentropy",
                  optimizer=OPTIM_C)

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


def load_X_y(videos_list, index, data_dir, ped_action_cats, batch_size=BATCH_SIZE):
    X = np.zeros((batch_size, VIDEO_LENGTH,) + IMG_SIZE)
    y = []
    for i in range(batch_size):
        y_per_vid = []
        for j in range(VIDEO_LENGTH):
            frame_number = (videos_list[(index*batch_size + i), j])
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


def get_action_classes(action_labels, mode='softmax'):
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
                print ("Unknown action in labels. Exiting.")
                print (action)
                exit(0)
            if action.lower() == 'crossing':
                ped_action = simple_ped_set.index('crossing')
                simple_ped_actions_per_frame.append(ped_action)
            # if action.lower() == 'standing':
            #     ped_action = simple_ped_set.index('standing')
            #     simple_ped_actions_per_frame.append(ped_action)
            # if action.lower() == 'no ped':
            #     ped_action = simple_ped_set.index('no ped')
            #     simple_ped_actions_per_frame.append(ped_action)

        if mode=='softmax':
            if 2 in simple_ped_actions_per_frame:
                act = 2
            if 0 in simple_ped_actions_per_frame:
                act = 0
            if 1 in simple_ped_actions_per_frame:
                act = 1

            encoded_ped_action = to_categorical(act, len(simple_ped_set))
            count[act] = count[act] + 1

        elif mode=='sigmoid':
            for action in simple_ped_actions_per_frame:
                count[action] = count[action] + 1
                # Add all unique categorical one-hot vectors
                encoded_ped_action = encoded_ped_action + to_categorical(action, len(simple_ped_set))

        else:
            print ("No mode selected to determine action labels. Exiting.")
            exit(0)

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


def get_sklearn_metrics(y_true, y_pred, avg=None, pos_label=1):
    return precision_recall_fscore_support(y_true, np.round(y_pred), average=avg, pos_label=pos_label)


def get_classification_report(y_true, y_pred):
    return classification_report(y_true, np.round(y_pred), target_names=['crossing', 'not crossing'])


def train(BATCH_SIZE, ENC_WEIGHTS, DEC_WEIGHTS, CLA_WEIGHTS):
    print("Loading data definitions.")

    frames_source = hkl.load(os.path.join(DATA_DIR, 'sources_train_208.hkl'))
    videos_list_1 = get_video_lists(frames_source=frames_source, stride=8, frame_skip=0)
    videos_list_2 = get_video_lists(frames_source=frames_source, stride=8, frame_skip=1)
    videos_list = np.concatenate((videos_list_1, videos_list_2), axis=0)

    # Load actions from annotations
    action_labels = hkl.load(os.path.join(DATA_DIR, 'annotations_train_208.hkl'))
    ped_action_classes, ped_class_count = get_action_classes(action_labels=action_labels, mode='sigmoid')
    print("Training Stats: " + str(ped_class_count))

    # videos_list = remove_zero_classes(videos_list, ped_action_classes)
    # classwise_videos_list, count = get_classwise_data(videos_list, ped_action_classes)
    # videos_list = prob_subsample(classwise_videos_list, count)

    if RAM_DECIMATE:
        frames = load_to_RAM(frames_source=frames_source)

    if SHUFFLE:
        # Shuffle images to aid generalization
        videos_list = np.random.permutation(videos_list)

    # Setup validation
    val_frames_source = hkl.load(os.path.join(VAL_DATA_DIR, 'sources_val_208.hkl'))
    val_videos_list = get_video_lists(frames_source=val_frames_source, stride=8, frame_skip=0)
    # Load val action annotations
    val_action_labels = hkl.load(os.path.join(VAL_DATA_DIR, 'annotations_val_208.hkl'))
    val_ped_action_classes, val_ped_class_count = get_action_classes(val_action_labels, mode='sigmoid')
    # val_videos_list = remove_zero_classes(val_videos_list, val_ped_action_classes)
    print("Val Stats: " + str(val_ped_class_count))

    # Build the Spatio-temporal Autoencoder
    print ("Creating models.")
    # Build stacked classifier
    # classifier = pretrained_c3d()
    classifier = ensemble_c3d()
    # classifier = c3d_scratch()
    classifier.compile(loss="binary_crossentropy",
                       optimizer=OPTIM_C,
                       # metrics=[metric_precision, metric_recall, metric_mpca, 'accuracy'])
                       metrics=['acc'])

    # Build attention layer output
    intermediate_classifier = Model(inputs=classifier.layers[0].input, outputs=classifier.layers[1].output)
    mask_gen_1 = Sequential()
    # mask_gen_1.add(encoder)
    mask_gen_1.add(intermediate_classifier)
    mask_gen_1.compile(loss='binary_crossentropy', optimizer=OPTIM_C)

    run_utilities(classifier, CLA_WEIGHTS)

    n_videos = videos_list.shape[0]
    n_val_videos = val_videos_list.shape[0]

    NB_ITERATIONS = int(n_videos/BATCH_SIZE)
    # NB_ITERATIONS = 5
    NB_VAL_ITERATIONS = int(n_val_videos/BATCH_SIZE)
    # NB_VAL_ITERATIONS = 5

    # Setup TensorBoard Callback
    TC_cla = tb_callback.TensorBoard(log_dir=TF_LOG_CLA_DIR, histogram_freq=0, write_graph=False, write_images=False)
    LRS_clas = lrs_callback.LearningRateScheduler(schedule=schedule)
    LRS_clas.set_model(classifier)


    print ("Beginning Training.")
    # Begin Training
    # Train Classifier
    if CLASSIFIER:
        print("Training Classifier...")
        for epoch in range(1, NB_EPOCHS_CLASS+1):
            print("\n\nEpoch ", epoch)
            c_loss = []
            val_c_loss = []

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
                    # videos_list = prob_subsample(classwise_videos_list, count)
                    X, y = load_X_y_RAM(videos_list, index, frames, ped_action_classes)
                else:
                    # videos_list = prob_subsample(classwise_videos_list, count)
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

                        label_true = str(y_orig_classes[k, j])
                        label_pred = str([round(float(i), 2) for i in ped_pred_class[k]])

                        cv2.putText(pred_seq, 'truth: ' + label_true,
                                    (2 + j * (208), 94 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        cv2.putText(pred_seq, label_pred,
                                    (2 + j * (208), 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)

                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_cla_pred.png"), pred_seq)

                slices = mask_gen_1.predict(X_train)
                slice_images = arrange_images(slices)
                slice_images = slice_images * 127.5 + 127.5
                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_slice_pred.png"), slice_images)

            # Run over val data
            print('')
            y_val_pred = []
            y_val_true = []
            for index in range(NB_VAL_ITERATIONS):
                X, y = load_X_y(val_videos_list, index, VAL_DATA_DIR, val_ped_action_classes)
                X_val = X
                y_true_class = y[:, CLASS_TARGET_INDEX]

                val_c_loss.append(classifier.test_on_batch(X_val, y_true_class))
                y_val_true.extend(y_true_class)
                y_val_pred.extend(classifier.predict(X_val, verbose=0))

                arrow = int(index / (NB_VAL_ITERATIONS / 40))
                stdout.write("\rIter: " + str(index) + "/" + str(NB_VAL_ITERATIONS - 1) + "  " +
                             "val_c_loss: " +  str([ val_c_loss[len(val_c_loss) - 1][j]  for j in [0, 1]]))
                stdout.flush()

            if SAVE_GENERATED_IMAGES:
                # Save generated images to file
                val_ped_pred_class = classifier.predict(X_val, verbose=0)
                # pred_seq = arrange_images(np.concatenate((X_train, predicted_images), axis=1))
                pred_seq = arrange_images(X_val)
                pred_seq = pred_seq * 127.5 + 127.5

                font = cv2.FONT_HERSHEY_SIMPLEX
                y_orig_classes = y
                # Add labels as text to the image

                for k in range(BATCH_SIZE):
                    for j in range(int(VIDEO_LENGTH)):
                        class_num_past = np.argmax(y_orig_classes[k, j])
                        class_num_y = np.argmax(val_ped_pred_class[k])

                        label_true = str(y_orig_classes[k, j])
                        label_pred = str([round(float(i), 2) for i in ped_pred_class[k]])


                        cv2.putText(pred_seq, 'truth: ' + label_true,
                                    (2 + j * (208), 94 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        cv2.putText(pred_seq, label_pred,
                                    (2 + j * (208), 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)

                cv2.imwrite(os.path.join(CLA_GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_cla_val_pred.png"), pred_seq)

            # then after each epoch
            avg_c_loss = np.mean(np.asarray(c_loss, dtype=np.float32), axis=0)
            avg_val_c_loss = np.mean(np.asarray(val_c_loss, dtype=np.float32), axis=0)

            train_prec, train_rec, train_fbeta, train_support = get_sklearn_metrics(np.asarray(y_train_true),
                                                                                    np.asarray(y_train_pred),
                                                                                    avg='binary',
                                                                                    pos_label=1)
            val_prec, val_rec, val_fbeta, val_support = get_sklearn_metrics(np.asarray(y_val_true),
                                                                                np.asarray(y_val_pred),
                                                                                avg='binary',
                                                                                pos_label=1)

            loss_values = np.asarray(avg_c_loss.tolist() + [train_prec.tolist()] +
                                     [train_rec.tolist()] +
                                     avg_val_c_loss.tolist() + [val_prec.tolist()] +
                                     [val_rec.tolist()], dtype=np.float32)

            precs = ['prec_' + action for action in simple_ped_set]
            recs = ['rec_' + action for action in simple_ped_set]
            fbeta = ['fbeta_' + action for action in simple_ped_set]
            c_loss_keys = ['c_' + metric for metric in classifier.metrics_names+precs+recs]
            val_c_loss_keys = ['c_val_' + metric for metric in classifier.metrics_names+precs+recs]

            loss_keys = c_loss_keys + val_c_loss_keys
            logs = dict(zip(loss_keys, loss_values))

            TC_cla.on_epoch_end(epoch, logs)

            # Log the losses
            with open(os.path.join(LOG_DIR, 'losses_cla.json'), 'a') as log_file:
                log_file.write("{\"epoch\":%d, %s\n" % (epoch, str(logs).strip('{')))

            print("\nAvg c_loss: " + str(avg_c_loss) +
                  " Avg val_c_loss: " + str(avg_val_c_loss))

            print ("Train Prec: %.2f, Recall: %.2f, Fbeta: %.2f" %(train_prec, train_rec, train_fbeta))
            print("Val Prec: %.2f, Recall: %.2f, Fbeta: %.2f" % (val_prec, val_rec, val_fbeta))

            # Save model weights per epoch to file
            classifier.save_weights(os.path.join(CHECKPOINT_DIR, 'classifier_cla_epoch_' + str(epoch) + '.h5'),
                                    True)
            classifier.save(os.path.join(CHECKPOINT_DIR, 'full_classifier_cla_epoch_' + str(epoch) + '.h5'))


        print (get_classification_report(np.asarray(y_train_true), np.asarray(y_train_pred)))
        print (get_classification_report(np.asarray(y_val_true), np.asarray(y_val_pred)))


def test(CLA_WEIGHTS):

    if not os.path.exists(TEST_RESULTS_DIR + '/pred/'):
        os.mkdir(TEST_RESULTS_DIR + '/pred/')

    # Setup test
    test_frames_source = hkl.load(os.path.join(TEST_DATA_DIR, 'sources_test_208.hkl'))
    # test_videos_list = get_video_lists(frames_source=test_frames_source, stride=8, frame_skip=0)
    test_videos_list = get_video_lists(frames_source=test_frames_source, stride=16, frame_skip=0)
    # test_videos_list = get_video_lists(frames_source=test_frames_source, stride=16, frame_skip=2)
    # Load test action annotations
    test_action_labels = hkl.load(os.path.join(TEST_DATA_DIR, 'annotations_test_208.hkl'))
    test_ped_action_classes, test_ped_class_count = get_action_classes(test_action_labels, mode='sigmoid')
    print("Test Stats: " + str(test_ped_class_count))

    # Build the Spatio-temporal Autoencoder
    print("Creating models.")
    # Build stacked classifier
    # classifier = pretrained_c3d()
    classifier = ensemble_c3d()
    # classifier = c3d_scratch()
    classifier.compile(loss="binary_crossentropy",
                       optimizer=OPTIM_C,
                       # metrics=[metric_precision, metric_recall, metric_mpca, 'accuracy'])
                       metrics=['acc'])

    run_utilities(classifier, CLA_WEIGHTS)

    n_test_videos = test_videos_list.shape[0]

    NB_TEST_ITERATIONS = int(n_test_videos / TEST_BATCH_SIZE)
    # NB_TEST_ITERATIONS = 5

    # Setup TensorBoard Callback
    TC_cla = tb_callback.TensorBoard(log_dir=TF_LOG_CLA_DIR, histogram_freq=0, write_graph=False, write_images=False)
    LRS_clas = lrs_callback.LearningRateScheduler(schedule=schedule)
    LRS_clas.set_model(classifier)

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
            X, y = load_X_y(test_videos_list, index, TEST_DATA_DIR, test_ped_action_classes, batch_size=TEST_BATCH_SIZE)
            X_test = X
            y_true_class = y[:, CLASS_TARGET_INDEX]

            iter_starttime.append(time.time())
            test_ped_pred_class = classifier.predict(X_test, verbose=0)
            iter_endtime.append(time.time())

            test_c_loss.append(classifier.test_on_batch(X_test, y_true_class))
            y_test_true.extend(y_true_class)
            y_test_pred.extend(classifier.predict(X_test, verbose=0))

            arrow = int(index / (NB_TEST_ITERATIONS / 40))
            stdout.write("\rIter: " + str(index) + "/" + str(NB_TEST_ITERATIONS - 1) + "  " +
                         "test_c_loss: " + str([test_c_loss[len(test_c_loss) - 1][j] for j in [0, 1]]))
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

                for k in range(TEST_BATCH_SIZE):
                    for j in range(int(VIDEO_LENGTH)):

                        if (y_orig_classes[k, j] > 0.5):
                            label_true = "crossing"
                        else:
                            label_true = "not crossing"

                        if (test_ped_pred_class[k] > 0.5):
                            label_pred = "crossing"
                        else:
                            label_pred = "not crossing"

                        cv2.putText(pred_seq, 'truth: ' + label_true,
                                    (2 + j * (208), 94 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        cv2.putText(pred_seq, label_pred,
                                    (2 + j * (208), 114 + k * 128), font, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)

                cv2.imwrite(os.path.join(TEST_RESULTS_DIR + '/pred/', str(index) + "_cla_test_pred.png"),
                            pred_seq)

        # then after each epoch
        avg_test_c_loss = np.mean(np.asarray(test_c_loss, dtype=np.float32), axis=0)
        test_prec, test_rec, test_fbeta, test_support = get_sklearn_metrics(np.asarray(y_test_true),
                                                                            np.asarray(y_test_pred),
                                                                            avg='binary',
                                                                            pos_label=1)
        print("\nAvg test_c_loss: " + str(avg_test_c_loss))
        print("Test Prec: %.4f, Recall: %.4f, Fbeta: %.4f" % (test_prec, test_rec, test_fbeta))

        test_acc = accuracy_score(y_test_true, np.round(y_test_pred))
        print("Test Accuracy: %.4f" % (test_acc))

        avg_prec = average_precision_score(y_test_true, y_test_pred)
        print("Average precision: %.4f" % (avg_prec))

        precisions, recalls, thresholds = precision_recall_curve(y_test_true, y_test_pred)
        print("PR curve precisions: "  + str(precisions))
        print("PR curve recalls: " + str(recalls))
        print("PR curve thresholds: " + str(thresholds))
        print("PR curve prec mean: %.4f" %(np.mean(precisions)))
        print("PR curve prec std: %.4f" %(np.std(precisions)))
        print("Number of thresholds: %.4f" %(len(thresholds)))

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


def test_mtcp(CLA_WEIGHTS):

    if not os.path.exists(TEST_RESULTS_DIR + '/pred/'):
        os.mkdir(TEST_RESULTS_DIR + '/pred/')

    # Setup test
    test_frames_source = hkl.load(os.path.join(TEST_DATA_DIR, 'sources_test_208.hkl'))
    # test_videos_list = get_video_lists(frames_source=test_frames_source, stride=8, frame_skip=0)
    test_videos_list = get_video_lists(frames_source=test_frames_source, stride=16, frame_skip=0)
    # test_videos_list = get_video_lists(frames_source=test_frames_source, stride=16, frame_skip=2)
    # Load test action annotations
    test_action_labels = hkl.load(os.path.join(TEST_DATA_DIR, 'annotations_test_208.hkl'))
    test_ped_action_classes, test_ped_class_count = get_action_classes(test_action_labels, mode='sigmoid')
    print("Test Stats: " + str(test_ped_class_count))

    # Build the Spatio-temporal Autoencoder
    print("Creating models.")
    # Build stacked classifier
    # classifier = pretrained_c3d()
    classifier = ensemble_c3d()
    # classifier = c3d_scratch()
    classifier.compile(loss="binary_crossentropy",
                       optimizer=OPTIM_C,
                       # metrics=[metric_precision, metric_recall, metric_mpca, 'accuracy'])
                       metrics=['acc'])

    run_utilities(classifier, CLA_WEIGHTS)

    n_test_videos = test_videos_list.shape[0]

    NB_TEST_ITERATIONS = int(n_test_videos / TEST_BATCH_SIZE)
    # NB_TEST_ITERATIONS = 5

    # Setup TensorBoard Callback
    TC_cla = tb_callback.TensorBoard(log_dir=TF_LOG_CLA_DIR, histogram_freq=0, write_graph=False,
                                     write_images=False)
    LRS_clas = lrs_callback.LearningRateScheduler(schedule=schedule)
    LRS_clas.set_model(classifier)
    if CLASSIFIER:
        print("Testing Classifier...")
        # Run over test data
        print('')
        # Time to correct prediction
        tcp_list = []
        tcp_true_list = []
        tcp_pred_list = []
        y_test_pred = []
        y_test_true = []
        test_c_loss = []
        index = 0
        tcp = 1
        while index < NB_TEST_ITERATIONS:
            X, y = load_X_y(test_videos_list, index, TEST_DATA_DIR, test_ped_action_classes,
                            batch_size=TEST_BATCH_SIZE)

            y_past_class = y[:, 0]
            y_end_class = y[:,-1]

            if y_end_class[0] == y_past_class[0]:
                index = index + 1
                continue
            else:
                stdout.write("\rIter: " + str(index) + "/" + str(NB_TEST_ITERATIONS - 1))
                stdout.flush()
                for fnum in range (int(VIDEO_LENGTH/2) + 1):

                    X, y = load_X_y(test_videos_list, index, TEST_DATA_DIR, test_ped_action_classes,
                                    batch_size=TEST_BATCH_SIZE)
                    X_test = X

                    y_true_imgs = X[:, int(VIDEO_LENGTH / 2):]
                    y_true_class = y[:, VIDEO_LENGTH - fnum - 1]
                    if y[:, 0] == y_true_class[0]:
                        break

                    if (fnum + 1 > 16):
                        tcp_pred_list.append(y_pred_class[0])
                        tcp_true_list.append(y_true_class[0])
                        break

                    y_pred_class = classifier.predict(X_test, verbose=0)
                    y_test_pred.extend(classifier.predict(X_test, verbose=0))
                    test_c_loss.append(classifier.test_on_batch(X_test, y_true_class))
                    y_test_true.extend(y_true_class)

                    test_ped_pred_class = classifier.predict(X_test, verbose=0)
                    # pred_seq = arrange_images(np.concatenate((X_train, predicted_images), axis=1))
                    pred_seq = arrange_images(X_test)
                    pred_seq = pred_seq * 127.5 + 127.5

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
                        tcp_pred_list.append(y_pred_class[0])
                        tcp_true_list.append(y_true_class[0])
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

        print ("-------------------------------------------")
        print ("Test cases where there is a change in label")

        test_prec, test_rec, test_fbeta, test_support = get_sklearn_metrics(np.asarray(tcp_true_list),
                                                                            np.asarray(tcp_pred_list),
                                                                            avg='binary',
                                                                            pos_label=1)
        print("Test Prec: %.4f, Recall: %.4f, Fbeta: %.4f" % (test_prec, test_rec, test_fbeta))

        test_acc = accuracy_score(tcp_true_list, np.round(tcp_pred_list))
        print("Test Accuracy: %.4f" % (test_acc))

        avg_prec = average_precision_score(tcp_true_list, tcp_pred_list)
        print("Average precision: %.4f" % (avg_prec))

        precisions, recalls, thresholds = precision_recall_curve(tcp_true_list, tcp_pred_list)
        print("PR curve precisions: " + str(precisions))
        print("PR curve recalls: " + str(recalls))
        print("PR curve thresholds: " + str(thresholds))
        print("PR curve prec mean: %.4f" % (np.mean(precisions)))
        print("PR curve prec std: %.4f" % (np.std(precisions)))
        print("Number of thresholds: %.4f" % (len(thresholds)))

        print("Classification Report")
        print(get_classification_report(np.asarray(tcp_true_list), np.asarray(tcp_pred_list)))

        print("Confusion matrix")
        tn, fp, fn, tp = confusion_matrix(tcp_true_list, np.round(tcp_pred_list)).ravel()
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
        test(CLA_WEIGHTS=args.cla_weights)

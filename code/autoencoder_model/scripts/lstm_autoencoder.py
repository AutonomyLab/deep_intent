from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hickle as hkl
import numpy as np

np.random.seed(2 ** 10)
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers.core import Activation
from keras.utils.vis_utils import plot_model
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import RepeatVector
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers.core import Reshape
from keras.callbacks import LearningRateScheduler
from keras.layers.advanced_activations import LeakyReLU
from config_la import *

import tb_callback
import lrs_callback
import argparse
import math
import os
import cv2
from sys import stdout


def encoder_model():
    model = Sequential()

    # 128x128
    model.add(TimeDistributed(Conv2D(filters=256,
                                     kernel_size=(5, 5),
                                     dilation_rate=(1, 1),
                                     strides=(4, 4),
                                     padding='same'),
                                     input_shape=(VIDEO_LENGTH, 128, 128, 3)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.5)))

    # 32x32
    model.add(TimeDistributed(Conv2D(filters=256,
                                     kernel_size=(5, 5),
                                     dilation_rate=(2, 2),
                                     strides=(1, 1),
                                     padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.5)))

    # 32x32
    model.add(TimeDistributed(Conv2D(filters=128,
                                     kernel_size=(5, 5),
                                     dilation_rate=(3, 3),
                                     strides=(1, 1),
                                     padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.5)))

    # 32x32
    model.add(TimeDistributed(Conv2D(filters=64,
                                     kernel_size=(5, 5),
                                     dilation_rate=(4, 4),
                                     strides=(1, 1),
                                     padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.5)))

    return model


def temporal_model():
    model = Sequential()

    # 32x32
    model.add(ConvLSTM2D(filters=64,
                         kernel_size=(3, 3),
                         padding='same',
                         strides=(1, 1),
                         return_sequences=True,
                         # activation='tanh',
                         dropout=0.5,
                         recurrent_dropout=0.5,
                         input_shape=(VIDEO_LENGTH, 32, 32, 64)))
    model.add(TimeDistributed(BatchNormalization()))

    # 16x16
    model.add(ConvLSTM2D(filters=64,
                         kernel_size=(3, 3),
                         padding='same',
                         strides=(1, 1),
                         return_sequences=True,
                         # activation='tanh',
                         dropout=0.5,
                         recurrent_dropout=0.5))
    model.add(TimeDistributed(BatchNormalization()))

    # 16x16
    model.add(ConvLSTM2D(filters=64,
                         kernel_size=(3, 3),
                         padding='same',
                         strides=(1, 1),
                         return_sequences=True,
                         # activation='tanh',
                         dropout=0.5,
                         recurrent_dropout = 0.5))
    model.add(TimeDistributed(BatchNormalization()))

    # model.add(Flatten())
    # model.add(RepeatVector(n=VIDEO_LENGTH))
    # model.add(Reshape(target_shape=(VIDEO_LENGTH, 32, 32, 64)))

    return model


def decoder_model():
    model = Sequential()

    # 16x16
    model.add(TimeDistributed(Conv2DTranspose(filters=128,
                                              kernel_size=(5, 5),
                                              dilation_rate=(4, 4),
                                              padding='same',
                                              strides=(1, 1)),
                                              input_shape=(VIDEO_LENGTH, 32, 32, 64)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    # model.add(TimeDistributed(LeakyReLU(0.2)))
    model.add(TimeDistributed(Dropout(0.5)))

    model.add(TimeDistributed(Conv2DTranspose(filters=256,
                                              kernel_size=(5, 5),
                                              dilation_rate=(3, 3),
                                              padding='same',
                                              strides=(1, 1))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    # model.add(TimeDistributed(LeakyReLU(0.2)))
    model.add(TimeDistributed(Dropout(0.5)))

    model.add(TimeDistributed(Conv2DTranspose(filters=256,
                                              kernel_size=(5, 5),
                                              dilation_rate=(2, 2),
                                              padding='same',
                                              strides=(1, 1))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    # model.add(TimeDistributed(LeakyReLU(0.2)))
    model.add(TimeDistributed(Dropout(0.5)))

    # 64x64
    model.add(TimeDistributed(Conv2DTranspose(filters=3,
                                              kernel_size=(5, 5),
                                              dilation_rate=(1, 1),
                                              strides=(4, 4),
                                              padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('tanh')))
    model.add(TimeDistributed(Dropout(0.5)))

    return model

def set_trainability(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def autoencoder_model(encoder, temporizer, decoder):
    model = Sequential()
    model.add(encoder)
    model.add(temporizer)
    model.add(decoder)
    return model


def combine_images(generated_images, X):
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

    return orig_image, image


def load_weights(weights_file, model):
    model.load_weights(weights_file)


def run_utilities(encoder, temporizer, decoder, autoencoder, ENC_WEIGHTS, TEM_WEIGHTS, DEC_WEIGHTS):
    if PRINT_MODEL_SUMMARY:
        print (encoder.summary())
        print (temporizer.summary())
        print (decoder.summary())
        print (autoencoder.summary())
        # exit(0)

    # Save model to file
    if SAVE_MODEL:
        print ("Saving models to file...")
        model_json = encoder.to_json()
        with open(os.path.join(MODEL_DIR, "encoder.json"), "w") as json_file:
            json_file.write(model_json)
        plot_model(encoder, to_file=os.path.join(MODEL_DIR, 'encoder.png'), show_shapes=True)

        with open(os.path.join(MODEL_DIR, "temporizer.json"), "w") as json_file:
            json_file.write(model_json)
        plot_model(temporizer, to_file=os.path.join(MODEL_DIR, 'temporizer.png'), show_shapes=True)

        with open(os.path.join(MODEL_DIR, "decoder.json"), "w") as json_file:
            json_file.write(model_json)
        plot_model(decoder, to_file=os.path.join(MODEL_DIR, 'decoder.png'), show_shapes=True)

        model_json = autoencoder.to_json()
        with open(os.path.join(MODEL_DIR, "autoencoder.json"), "w") as json_file:
            json_file.write(model_json)
        plot_model(autoencoder, to_file=os.path.join(MODEL_DIR, 'autoencoder.png'), show_shapes=True)

    if ENC_WEIGHTS != "None":
        print ("Pre-loading encoder with weights...")
        load_weights(ENC_WEIGHTS, encoder)
    if TEM_WEIGHTS != "None":
        print ("Pre-loading temporizer with weights...")
        load_weights(TEM_WEIGHTS, temporizer)
    if DEC_WEIGHTS != "None":
        print ("Pre-loading decoder with weights...")
        load_weights(DEC_WEIGHTS, decoder)


def load_X(videos_list, index, data_dir):
    X = np.zeros((BATCH_SIZE, VIDEO_LENGTH,) + IMG_SIZE)
    for i in range(BATCH_SIZE):
        for j in range(VIDEO_LENGTH):
            filename = "frame_" + str(videos_list[(index*BATCH_SIZE + i), j]) + ".png"
            im_file = os.path.join(data_dir, filename)
            try:
                frame = cv2.imread(im_file, cv2.IMREAD_COLOR)
                X[i, j] = (frame.astype(np.float32) - 127.5) / 127.5
            except AttributeError as e:
                print (im_file)
                print(e)

    return X


def train(BATCH_SIZE, ENC_WEIGHTS, TEM_WEIGHTS, DEC_WEIGHTS):
    print ("Loading data...")
    frames_source = hkl.load(os.path.join(DATA_DIR, 'sources_train_128.hkl'))

    # Build video progressions
    videos_list = []
    start_frame_index = 1
    end_frame_index = VIDEO_LENGTH + 1
    while (end_frame_index <= len(frames_source)):
        frame_list = frames_source[start_frame_index:end_frame_index]
        if (len(set(frame_list)) == 1):
            videos_list.append(range(start_frame_index, end_frame_index))
            start_frame_index = start_frame_index + 1
            end_frame_index = end_frame_index + 1
        else:
            start_frame_index = end_frame_index - 1
            end_frame_index = start_frame_index + VIDEO_LENGTH

    videos_list = np.asarray(videos_list, dtype=np.int32)
    n_videos = videos_list.shape[0]

    if SHUFFLE:
        # Shuffle images to aid generalization
        videos_list = np.random.permutation(videos_list)

    # Build video progressions
    videos_list = []
    start_frame_index = 1
    end_frame_index = VIDEO_LENGTH + 1
    while (end_frame_index <= len(frames_source)):
        frame_list = frames_source[start_frame_index:end_frame_index]
        if (len(set(frame_list)) == 1):
            videos_list.append(range(start_frame_index, end_frame_index))
            start_frame_index = start_frame_index + 1
            end_frame_index = end_frame_index + 1
        else:
            start_frame_index = end_frame_index - 1
            end_frame_index = start_frame_index + VIDEO_LENGTH

    videos_list = np.asarray(videos_list, dtype=np.int32)
    n_videos = videos_list.shape[0]


    # Build the Spatio-temporal Autoencoder
    print ("Creating models...")
    encoder = encoder_model()
    temporizer = temporal_model()
    decoder = decoder_model()

    # print (encoder.summary())
    # print (temporizer.summary())
    # print (decoder.summary())

    autoencoder = autoencoder_model(encoder, temporizer, decoder)
    run_utilities(encoder, temporizer, decoder, autoencoder, ENC_WEIGHTS, TEM_WEIGHTS, DEC_WEIGHTS)

    autoencoder.compile(loss='mean_squared_error', optimizer=OPTIM)

    NB_ITERATIONS = int(n_videos/BATCH_SIZE)

    # Setup TensorBoard Callback
    TC = tb_callback.TensorBoard(log_dir=TF_LOG_DIR, histogram_freq=0, write_graph=False, write_images=False)
    LRS = lrs_callback.LearningRateScheduler(schedule=schedule)
    LRS.set_model(autoencoder)

    print ("Beginning Training...")
    # Begin Training
    for epoch in range(NB_EPOCHS):
        print("\n\nEpoch ", epoch)
        loss = []

        # Set learning rate every epoch
        LRS.on_epoch_begin(epoch=epoch)
        lr = K.get_value(autoencoder.optimizer.lr)
        print ("Learning rate: " + str(lr))

        for index in range(NB_ITERATIONS):
            # Train Autoencoder
            X = load_X(videos_list, index, DATA_DIR)
            loss.append(autoencoder.train_on_batch(X, X))

            arrow = int(index / (NB_ITERATIONS / 40))
            stdout.write("\rIteration: " + str(index) + "/" + str(NB_ITERATIONS-1) + "  " +
                         "loss: " + str(loss[len(loss)-1]) +
                         "\t    [" + "{0}>".format("="*(arrow)))
            stdout.flush()

        if SAVE_GENERATED_IMAGES:
            # Save generated images to file
            generated_images = autoencoder.predict(X, verbose=0)
            orig_image, image = combine_images(generated_images, X)
            image = image * 127.5 + 127.5
            orig_image = orig_image * 127.5 + 127.5
            if epoch == 0 :
                cv2.imwrite(os.path.join(GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_orig.png"), orig_image)
            cv2.imwrite(os.path.join(GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + ".png"), image)

        # then after each epoch/iteration
        avg_loss = sum(loss)/len(loss)
        logs = {'loss': avg_loss}
        TC.on_epoch_end(epoch, logs)

        # Log the losses
        with open(os.path.join(LOG_DIR, 'losses.json'), 'a') as log_file:
            log_file.write("{\"epoch\":%d, \"loss\":%f};\n" % (epoch, avg_loss))

        print ("\nAvg loss: " +  str(avg_loss))

        # Save model weights per epoch to file
        encoder.save_weights(os.path.join(CHECKPOINT_DIR, 'encoder_epoch_'+str(epoch)+'.h5'), True)
        temporizer.save_weights(os.path.join(CHECKPOINT_DIR, 'temporizer_epoch_' + str(epoch) + '.h5'), True)
        decoder.save_weights(os.path.join(CHECKPOINT_DIR, 'decoder_epoch_' + str(epoch) + '.h5'), True)

    # End TensorBoard Callback
    TC.on_train_end('_')


def test(ENC_WEIGHTS, TEM_WEIGHTS, DEC_WEIGHTS):
    # Create test results directory
    TEST_RESULTS_DIR = os.path.join(TEST_DATA_DIR, "../test_results/")
    if not os.path.exists(TEST_RESULTS_DIR):
        os.mkdir(TEST_RESULTS_DIR)

    # Create models
    print ("Creating models...")
    encoder = encoder_model()
    temporizer = temporal_model()
    decoder = decoder_model()
    autoencoder = autoencoder_model(encoder, temporizer, decoder)

    run_utilities(encoder, temporizer, decoder, autoencoder, ENC_WEIGHTS, TEM_WEIGHTS, DEC_WEIGHTS)
    autoencoder.compile(loss='mean_squared_error', optimizer=OPTIM)

    # Build video progressions
    frames_source = hkl.load(os.path.join(TEST_DATA_DIR, 'sources_test_128.hkl'))
    videos_list = []
    start_frame_index = 1
    end_frame_index = VIDEO_LENGTH + 1
    while (end_frame_index <= len(frames_source)):
        frame_list = frames_source[start_frame_index:end_frame_index]
        if (len(set(frame_list)) == 1):
            videos_list.append(range(start_frame_index, end_frame_index))
            start_frame_index = start_frame_index + 1
            end_frame_index = end_frame_index + 1
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
        X_test = load_X(videos_list, index, TEST_DATA_DIR)
        loss.append(autoencoder.test_on_batch(X_test, X_test))
        y = autoencoder.predict_on_batch(X_test)

        arrow = int(index / (NB_ITERATIONS / 40))
        stdout.write("\rIteration: " + str(index) + "/" + str(NB_ITERATIONS - 1) + "  " +
                     "loss: " + str(loss[len(loss) - 1]) +
                     "\t    [" + "{0}>".format("=" * (arrow)))
        stdout.flush()

        avg_loss = sum(loss) / len(loss)
        print ("\nAvg loss: " +  str(avg_loss))

        orig_image, image = combine_images(y, X_test)
        image = image * 127.5 + 127.5
        orig_image = orig_image * 127.5 + 127.5

        cv2.imwrite(os.path.join(TEST_RESULTS_DIR, str(index) + "_orig.png"), orig_image)
        cv2.imwrite(os.path.join(TEST_RESULTS_DIR, str(index) + ".png"), image)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--enc_weights", type=str, default="None")
    parser.add_argument("--tem_weights", type=str, default="None")
    parser.add_argument("--dec_weights", type=str, default="None")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size,
              ENC_WEIGHTS=args.enc_weights,
              TEM_WEIGHTS=args.tem_weights,
              DEC_WEIGHTS=args.dec_weights)

    if args.mode == "test":
        test(ENC_WEIGHTS=args.enc_weights,
             TEM_WEIGHTS=args.tem_weights,
             DEC_WEIGHTS=args.dec_weights)
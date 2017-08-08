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
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
# from keras.callbacks import TensorBoard
from scripts.config_fa import *

from scripts import tb_callback
import argparse
import math
import os
import cv2
from sys import stdout

def encoder_model():
    model = Sequential()

    # 64x64
    model.add(TimeDistributed(Conv2D(filters=64,
                                     input_shape=(2, 64, 64, 3),
                                     kernel_size=(4, 4),
                                     padding='same',
                                     strides=1)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.2)))

    # 64x64
    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=(4, 4), padding='valid', strides=2)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.2)))

    # 32x32
    model.add(TimeDistributed(Conv2D(filters=128, kernel_size=(4, 4), padding='valid', strides=2)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.2)))

    # 16x16
    model.add(TimeDistributed(Conv2D(filters=256, kernel_size=(4, 4), padding='valid', strides=2)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.2)))

    # 8x8
    model.add(TimeDistributed(Conv2D(filters=512, kernel_size=(4, 4), padding='valid', strides=2)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('tanh')))
    model.add(TimeDistributed(Dropout(0.2)))

    return model


def temporal_model():
    model = Sequential()
    model.add(ConvLSTM2D(filters=64,
                         kernel_size=(3, 3),
                         padding='same',
                         strides=1,
                         return_sequences=True,
                         input_size=(8, 8, 512)))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', strides=1, return_sequences=True))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', strides=1, return_sequences=True))
    model.add(TimeDistributed(BatchNormalization()))

    return model


def decoder_model():
    model = Sequential()

    # 8x8
    model.add(TimeDistributed(Conv2DTranspose(filters=256,
                                              kernel_size=(5, 5),
                                              padding='valid',
                                              strides=(2, 2),
                                              input_size=(8, 8, 512))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(LeakyReLU(0.2)))
    model.add(TimeDistributed(Dropout(0.5)))

    # 16x16
    model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(LeakyReLU(0.2)))
    model.add(TimeDistributed(Dropout(0.5)))

    # 32x32
    model.add(Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(LeakyReLU(0.2)))
    model.add(TimeDistributed(Dropout(0.5)))

    # # 64x64
    # model.add(Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    # model.add(TimeDistributed(BatchNormalization()))
    # model.add(TimeDistributed(LeakyReLU(0.2)))
    # model.add(TimeDistributed(Dropout(0.5)))

    # 64x64
    model.add(Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='tanh'))

    return model


def set_trainability(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def autoencoder_model(encoder, temporal_net, decoder):
    model = Sequential()
    model.add(encoder)
    model.add(temporal_net)
    model.add(decoder)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:]
    image = np.zeros((height * shape[0], width * shape[1], shape[2]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = img
    return image


def load_weights(weights_file, model):
    model.load_weights(weights_file)


def run_utilities(encoder, temporal_net, decoder, autoencoder, ENC_WEIGHTS, TEM_WEIGHTS, DEC_WEIGHTS):
    if PRINT_MODEL_SUMMARY:
        print (encoder.summary())
        print (temporal_net.summary())
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

        model_json = temporal_net.to_json()
        with open(os.path.join(MODEL_DIR, "temporal_net.json"), "w") as json_file:
            json_file.write(model_json)
        plot_model(temporal_net, to_file=os.path.join(MODEL_DIR, 'temporal_net.png'), show_shapes=True)

        model_json = decoder.to_json()
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
        print("Pre-loading temporal_net with weights...")
        load_weights(TEM_WEIGHTS, temporal_net)
    if DEC_WEIGHTS != "None":
        print ("Pre-loading decoder with weights...")
        load_weights(DEC_WEIGHTS, decoder)


def train(BATCH_SIZE, ENC_WEIGHTS, TEM_WEIGHTS, DEC_WEIGHTS):
    print ("Loading data...")
    X_train = hkl.load(os.path.join(DATA_DIR, 'X_train.hkl'))
    X_train = (X_train.astype(np.float32) - 127.5)/127.5

    if SHUFFLE:
        # Shuffle images to aid generalization
        X_train = np.random.permutation(X_train)

    # Build the Spatio-temporal Autoencoder
    print ("Creating model...")
    encoder = encoder_model()
    temporal_net = temporal_model()
    decoder = decoder_model()
    autoencoder = autoencoder_model(encoder, temporal_net, decoder)

    run_utilities(encoder, temporal_net, decoder, autoencoder, ENC_WEIGHTS, TEM_WEIGHTS, DEC_WEIGHTS)

    # generator.compile(loss='binary_crossentropy', optimizer='sgd')
    # GAN.compile(loss='binary_crossentropy', optimizer=g_optim)
    # set_trainability(discriminator, True)
    # discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    autoencoder.compile(loss='binary_crossentropy', optimizer=OPTIM)

    NB_ITERATIONS = int(X_train.shape[0]/BATCH_SIZE)

    # Setup TensorBoard Callback
    TC = tb_callback.TensorBoard(log_dir=TF_LOG_DIR, histogram_freq=0, write_graph=False, write_images=False)

    noise = np.zeros((BATCH_SIZE, 100), dtype=np.float32)

    print ("Beginning Training...")
    # Begin Training
    for epoch in range(NB_EPOCHS):
        print("\n\nEpoch ", epoch)
        loss = []

        Y = X_train[0: BATCH_SIZE]
        for index in range(NB_ITERATIONS - 1):
            # Train Autoencoder with future frames
            # Limit BATCH_SIZE to 10-20
            X = Y
            Y = X_train[(index + 1) * BATCH_SIZE: (index + 2) * BATCH_SIZE]

            loss.append(autoencoder.train_on_batch(X, Y))

            arrow = int(index/(NB_ITERATIONS/30))
            stdout.write("\rIteration: " + str(index) + "/" + str(NB_ITERATIONS-1) + "  " +
                         "loss: " + str(loss[len(loss)-1]) +
                         "\t    [" + "{0}>".format("="*(arrow)))
            stdout.flush()

        if SAVE_GENERATED_IMAGES:
            # Save generated images to file
            generated_images = autoencoder.predict(noise, verbose=0)
            image = combine_images(generated_images)
            image = image * 127.5 + 127.5
            cv2.imwrite(os.path.join(GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + ".png"), image)

        # then after each epoch/iteration
        avg_loss = sum(loss)/len(loss)
        logs = {'loss': loss}
        TC.on_epoch_end(epoch, logs)

        # Log the losses
        with open(os.path.join(LOG_DIR, 'losses.json'), 'a') as log_file:
            log_file.write("{\"epoch\":%d, \"d_loss\":%f};\n" % (epoch, avg_loss))

        # Save model weights per epoch to file
        encoder.save_weights(os.path.join(CHECKPOINT_DIR, 'encoder_epoch_'+str(epoch)+'.h5'), True)
        temporal_net.save_weights(os.path.join(CHECKPOINT_DIR, 'temporal_net_epoch_'+str(epoch)+'.h5'), True)
        decoder.save_weights(os.path.join(CHECKPOINT_DIR, 'decoder_epoch_' + str(epoch) + '.h5'), True)

    # End TensorBoard Callback
    TC.on_train_end(_)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--weights_enc", type=str, default="None")
    parser.add_argument("--weights_tem", type=str, default="None")
    parser.add_argument("--weights_dec", type=str, default="None")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size,
              ENC_WEIGHTS=args.weights_enc,
              TEM_WEIGHTS=args.weights_tem,
              DEC_WEIGHTS=args.weights_dec)
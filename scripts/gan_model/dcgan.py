from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import Input
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.utils.vis_utils import plot_model
# from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.datasets import mnist
from model_config import *
import hickle as hkl
import numpy as np
np.random.seed(2 ** 10)
from PIL import Image
import tb_callback
import argparse
import math
import os
import cv2
from sys import stdout
K.set_image_dim_ordering('tf')

def generator_model():

    # g_input = Input(shape=(100,))
    # x = Dense(512*4*4, input_dim=100)(g_input)
    # x = Activation('relu')(x)
    # x = Reshape((4, 4, 512), input_shape=(512*4*4,))(x)
    # x = Conv2D(filters=512, kernel_size=(4,4), padding='same')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    # x = BatchNormalization()(x)
    # x = (Activation('relu'))(x)
    # x = Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    # x = BatchNormalization()(x)
    # x = Activation('tanh')(x)
    # g_output = Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='tanh')(x)
    # model = Model(inputs=g_input, outputs=g_output)

    model = Sequential()
    model.add(Dense(input_dim=100, units=1024))
    model.add(Activation('tanh'))
    model.add(Dense(512*4*4))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    # model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Reshape((4, 4, 512), input_shape=(512*4*4,)))
    model.add(Conv2D(filters=512, kernel_size=(4,4), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    # model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    # model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2DTranspose(filters=3, kernel_size=(4,4), strides=(2,2), padding='same', activation='tanh'))

    return model


def discriminator_model():

    # d_input = Input(shape=IMAGE_SHAPE)
    # x = Conv2D(filters=64, kernel_size=(4, 4), padding='same', input_shape=IMAGE_SHAPE)(d_input)
    # x = LeakyReLU()(x)
    #
    # x = Conv2D(filters=64, kernel_size=(4, 4), strides=2, padding='same')(x)
    # x  = LeakyReLU()(x)
    #
    # x = Conv2D(filters=128, kernel_size=(4, 4), strides=2, padding='same')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU()(x)
    #
    # x = Conv2D(filters=256, kernel_size=(4, 4), strides=2, padding='same')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU()(x)
    #
    # x = Flatten()(x)
    # x = Dense(1024)(x)
    # x = Activation('tanh')(x)
    # x = Dense(1)(x)
    # d_output = Activation('sigmoid')(x)
    #
    # model = Model(inputs=d_input, outputs=d_output)

    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(4, 4), padding='same', input_shape=(64, 64, 3)))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=2, padding='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.5))

    model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=2, padding='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.5))

    model.add(Conv2D(filters=256, kernel_size=(4, 4), strides=2, padding='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.5))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def gan_model(generator, discriminator):
    # gan_input = Input(shape=(100,))
    # x = generator(gan_input)
    # set_trainability(discriminator, False)
    # gan_output = discriminator(x)
    #
    # model = Model(inputs=gan_input, outputs=gan_output)

    model = Sequential()
    model.add(generator)
    set_trainability(discriminator, False)
    model.add(discriminator)
    return model


def set_trainability(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


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


def train(BATCH_SIZE, GEN_WEIGHTS, DISC_WEIGHTS):
    print ("Loading data...")
    X_train = hkl.load(os.path.join(DATA_DIR, 'X_train.hkl'))
    X_train = (X_train.astype(np.float32) - 127.5)/127.5

    if SHUFFLE:
        # Shuffle images to aid generalization
        X_train = np.random.permutation(X_train)

    print ("Creating models...")
    # Create the Generator and Discriminator models
    generator = generator_model()
    discriminator = discriminator_model()

    # Create the full GAN model with discriminator non-trainable
    GAN = gan_model(generator, discriminator)
    d_optim = SGD(lr=0.001, momentum=0.5, nesterov=True)
    # g_optim = SGD(lr=0.0001, momentum=0.5, nesterov=True)
    # d_optim = Adam(lr=0.005, beta_1=0.5)
    g_optim = Adam(lr=0.0001, beta_1=0.5)

    generator.compile(loss='binary_crossentropy', optimizer='sgd')
    GAN.compile(loss='binary_crossentropy', optimizer=g_optim)
    set_trainability(discriminator, True)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    if PRINT_MODEL_SUMMARY:
        print (generator.summary())
        print (discriminator.summary())
        print (GAN.summary())

    # Save model to file
    if SAVE_MODEL:
        print ("Saving models to file...")
        model_json = generator.to_json()
        with open(os.path.join(MODEL_DIR, "generator.json"), "w") as json_file:
            json_file.write(model_json)
        plot_model(generator, to_file=os.path.join(MODEL_DIR, 'generator.png'), show_shapes=True)

        model_json = discriminator.to_json()
        with open(os.path.join(MODEL_DIR, "discriminator.json"), "w") as json_file:
            json_file.write(model_json)
        plot_model(discriminator, to_file=os.path.join(MODEL_DIR, 'discriminator.png'), show_shapes=True)

        model_json = GAN.to_json()
        with open(os.path.join(MODEL_DIR, "GAN.json"), "w") as json_file:
            json_file.write(model_json)
        plot_model(GAN, to_file=os.path.join(MODEL_DIR, 'GAN.png'), show_shapes=True)

    if GEN_WEIGHTS != "None":
        print ("Pre-loading generator with weights...")
        load_weights(GEN_WEIGHTS, generator)
    if DISC_WEIGHTS != "None":
        print ("Pre-loading discriminator with weights...")
        load_weights(DISC_WEIGHTS, discriminator)

    NB_ITERATIONS = int(X_train.shape[0]/BATCH_SIZE)

    # Setup TensorBoard Callback
    TC = tb_callback.TensorBoard(log_dir='/local_home/logs', histogram_freq=0, write_graph=False, write_images=False)
    # TC.set_model(generator, discriminator)

    noise = np.zeros((BATCH_SIZE, 100), dtype=np.float32)

    print ("Beginning Training...")
    # Begin Training
    for epoch in range(NB_EPOCHS):
        print("\n\nEpoch ", epoch)
        g_loss = []
        d_loss = []
        for index in range(NB_ITERATIONS):

            # Generate images
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = generator.predict(noise, verbose=0)

            # Train Discriminator
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss.append(discriminator.train_on_batch(X, y))
            # print("Epoch %d Batch %d d_loss : %f" % (epoch, index, d_loss))

            # Train GAN
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            set_trainability(discriminator, False)
            g_loss.append(GAN.train_on_batch(noise, [1] * BATCH_SIZE))
            set_trainability(discriminator, True)
            # print("Epoch %d Batch %d g_loss : %f" % (epoch, index, g_loss))

            arrow = int(index/10)
            stdout.write("\rIteration: " + str(index) + "/" + str(NB_ITERATIONS-1) + "  " +
                         "g_loss: " + str(g_loss[len(g_loss)-1]) + "\t    " + "d_loss: " + str(d_loss[len(g_loss)-1]) +
                         "\t    [" + "{0}>".format("="*(arrow)))
            stdout.flush()

        if SAVE_GENERATED_IMAGES:
            # Save generated images to file
            image = combine_images(generated_images)
            image = image * 127.5 + 127.5
            # Image.fromarray(image.astype(np.uint8)).save(str(epoch) + "_" + str(index) + ".png")
            cv2.imwrite(os.path.join(GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + ".png"), image)

        # then after each epoch/iteration
        avg_g_loss = sum(g_loss)/len(g_loss)
        avg_d_loss = sum(d_loss)/len(d_loss)
        logs = {'g_loss': avg_g_loss, 'd_loss': avg_d_loss}
        TC.on_epoch_end(epoch, logs)

        # Log the losses
        with open(os.path.join(LOG_DIR, 'losses.json'), 'a') as log_file:
            log_file.write("{\"epoch\":%d, \"g_loss\":%f, \"d_loss\":%f};\n" % (epoch, avg_g_loss, avg_d_loss))

        # Save model weights per epoch to file
        generator.save_weights(os.path.join(CHECKPOINT_DIR, 'generator_epoch_'+str(epoch)+'.h5'), True)
        discriminator.save_weights(os.path.join(CHECKPOINT_DIR, 'discriminator_epoch_'+str(epoch)+'.h5'), True)

    # End TensorBoard Callback
    TC.on_train_end(_)


def generate(BATCH_SIZE, nice=False):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator')
    if nice:
        discriminator = discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('discriminator')
        noise = np.zeros((BATCH_SIZE*20, 100))
        for i in range(BATCH_SIZE*20):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE, 1) +
                               (generated_images.shape[2:]), dtype=np.float32)
        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]
        image = combine_images(nice_images)
    else:
        noise = np.zeros((BATCH_SIZE, 100))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--weights_gen", type=str, default="None")
    parser.add_argument("--weights_disc", type=str, default="None")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size, GEN_WEIGHTS=args.weights_gen, DISC_WEIGHTS=args.weights_disc)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
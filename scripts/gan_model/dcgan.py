from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.utils.visualize_util import plot
# from keras.callbacks import TensorBoard
from keras.optimizers import SGD
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
K.set_image_dim_ordering('tf')

def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, units=1024))
    model.add(Activation('tanh'))
    model.add(Dense(512*4*4))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape((4, 4, 512), input_shape=(512*4*4,)))
    model.add(Conv2D(filters=512, kernel_size=(4,4), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add((Activation('relu')))
    model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add((Activation('relu')))
    # model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='tanh'))
    # model.add(UpSampling2D(size=(2, 2)))
    # model.add(Conv2D(filters=256, kernel_size=(4, 4), padding='same'))
    # model.add(Activation('tanh'))
    # model.add(UpSampling2D(size=(2, 2)))
    # model.add(Conv2D(filters=128, kernel_size=(4, 4), padding='same'))
    # model.add(Activation('tanh'))
    # model.add(UpSampling2D(size=(2, 2)))
    # model.add(Conv2D(filters=64, kernel_size=(4, 4), padding='same'))
    # model.add(Activation('tanh'))
    # model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add((Activation('tanh')))
    model.add(Conv2DTranspose(filters=3, kernel_size=(4,4), strides=(2,2), padding='same', activation='tanh'))
    # model.add(Conv2D(filters=3, kernel_size=(4, 4), padding='same'))
    # model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(4, 4), padding='same', input_shape=(64, 64, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(filters=256, kernel_size=(4, 4), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    set_trainability(discriminator, False)
    model.add(discriminator)
    return model


def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[2:]
    image = np.zeros((height*shape[0], width*shape[1]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[0, :, :]
    return image


def train(BATCH_SIZE):
    print ("Loading data...")
    X_train = hkl.load(os.path.join(DATA_DIR, 'X_train.hkl'))
    X_train = (X_train.astype(np.float32) - 127.5)/127.5

    print ("Creating models...")
    # Create the Generator and Discriminator models
    discriminator = discriminator_model()
    generator = generator_model()

    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)

    # Create the full GAN model with discriminator non-trainable
    generator_on_discriminator = generator_containing_discriminator(generator, discriminator)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator_on_discriminator.compile(loss='binary_crossentropy', optimizer=g_optim)
    set_trainability(discriminator, True)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    if PRINT_MODEL_SUMMARY:
        print (generator.summary())
        print (discriminator.summary())
        print (generator_on_discriminator.summary())

    # Save model to file
    if SAVE_MODEL:
        print ("Saving models to file...")
        model_json = generator.to_json()
        with open(os.path.join(MODEL_DIR, "generator.json"), "w") as json_file:
            json_file.write(model_json)
        plot(generator, to_file=os.path.join(MODEL_DIR, 'generator.png'), show_shapes=True)

        model_json = discriminator.to_json()
        with open(os.path.join(MODEL_DIR, "discriminator.json"), "w") as json_file:
            json_file.write(model_json)
        plot(discriminator, to_file=os.path.join(MODEL_DIR, 'discriminator.png'), show_shapes=True)

        model_json = generator_on_discriminator.to_json()
        with open(os.path.join(MODEL_DIR, "GAN.json"), "w") as json_file:
            json_file.write(model_json)
        plot(generator_on_discriminator, to_file=os.path.join(MODEL_DIR, 'GAN.png'), show_shapes=True)

    NB_ITERATIONS = int(X_train.shape[0]/BATCH_SIZE)

    # Setup TensorBoard Callback
    TC = tb_callback.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    TC.set_model(generator, discriminator)

    print ("Beginning Training...")
    # Begin Training
    for epoch in range(NB_EPOCHS):
        print("Epoch is", epoch)
        print("Number of batches", NB_ITERATIONS)
        for index in range(NB_ITERATIONS):

            # Generate images
            noise = np.random.uniform(-1, 1, size=[BATCH_SIZE, 100])
            print (noise.shape)
            image_batch = X_train[index*BATCH_SIZE : (index+1)*BATCH_SIZE]
            generated_images = generator.predict(noise, verbose=0)

            # Train Discriminator
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            print (X.shape)
            print (np.asarray(y).shape)
            d_loss = discriminator.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))

            print ("Discriminator Trained")
            # Train GAN
            noise = np.random.uniform(-1, 1, size=[BATCH_SIZE, 100])
            set_trainability(discriminator, False)
            g_loss = generator_on_discriminator.train_on_batch(noise, [1] * BATCH_SIZE)
            discriminator.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))

            # then after each epoch
            logs = {'g_loss': d_loss, 'd_loss': g_loss}
            TC.on_epoch_end(epoch, logs)

        if SAVE_GENERATED_IMAGES:
            # Save generated images to file
            image = combine_images(generated_images)
            image = image * 127.5 + 127.5
            Image.fromarray(image.astype(np.uint8)).save(str(epoch) + "_" + str(index) + ".png")

        # Save model weights per epoch to file
        generator.save_weights(os.path.join(CHECKPOINT_DIR, 'generator_epoch_'+str(epoch)+'.h5'), True)
        discriminator.save_weights(os.path.join(CHECKPOINT_DIR, 'discriminator_epoch_'+str(epoch)+'.h5'), True)

    # End TensorBoard Callback
    TC.on_train_end()


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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
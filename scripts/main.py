from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
np.random.seed(2 ** 10)
import os
import math
from keras.datasets import cifar10
from keras.models import Model, Sequential
from keras.layers import Input, Activation, merge, Dense, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import metrics
from keras import backend as K
K.set_image_dim_ordering('tf')

# -------------------------------------------------
# Background config:
data_path= '../data/'
model_path = '../models/'
checkpoint_path = '../checkpoints/'
print_model_summary = True
save_model_plot = True
data_augmentation = False

# -------------------------------------------------
def load_data():
    # Data configuration:
    print ("Loading data...")

    nb_classes = 10
    image_size = 32

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, Y_train, X_test, Y_test

# -------------------------------------------------

# Network configuration:
print ("Loading network/training configuration...")

batch_size = 128
nb_epochs = 200
lr_schedule = [60, 120, 160]  # epoch_step

original_x_dim = 28
original_y_dim = 28
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0

def schedule(epoch_idx):
    if (epoch_idx + 1) < lr_schedule[0]:
        return 0.1
    elif (epoch_idx + 1) < lr_schedule[1]:
        return 0.02  # lr_decay_ratio = 0.2
    elif (epoch_idx + 1) < lr_schedule[2]:
        return 0.004
    return 0.0008


sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)

# -------------------------------------------------
def create_model():
    print ("Creating model...")
    input_img = Input(shape=(original_x_dim, original_y_dim, 3))
    x = Convolution2D(64, 3, 3, border_mode='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)

    x = Convolution2D(64, 3, 3, border_mode='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)

    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    def vae_loss(x, x_decoded_mean):
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    vae = Model(x, x_decoded_mean)
    vae.compile(optimizer='rmsprop', loss=vae_loss)

    return model

# -------------------------------------------------
if __name__ == '__main__':
    model = create_model()
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])

    print (data_path)

    callbacks = [LearningRateScheduler(schedule=schedule),
                 ModelCheckpoint(checkpoint_path + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
                 ]

    print ("Saving model")
    with open(os.path.join(model_path, 'paintgan.json')) as f:
        f.write(model.to_json())

    if print_model_summary:
        print ("Model summary...")
        print (model.summary())

    if save_model_plot:
        print ("Saving model plot...")
        from keras.utils.visualize_util import plot
        plot(model, to_file=os.path.join(model_path, 'paintgan.png'), show_shapes=True)

    if data_augmentation:
        # Data augmentation if corresponding bool parameter set true
        print ("Using real-time data augmentation")

        train_datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            zca_whitening=True,
            horizontal_flip=True)

        train_datagen.fit(X_train)

        print ("Running training...")
        # fit the model on the batches generated by train_datagen.flow()
        model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epochs,
                            validation_data=train_datagen.flow(X_test, Y_test, batch_size=batch_size),
                            nb_val_samples=X_test.shape[0],
                            callbacks=callbacks)

    else:
        # Not using data augmentation
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epochs,
                  validation_data=(X_test, Y_test),
                  callbacks=callbacks,
                  shuffle=True)
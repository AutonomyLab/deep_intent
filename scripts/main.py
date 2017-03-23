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
from keras.layers.convolutional import Conv2D
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

# input image dimensions
img_rows, img_cols, img_chns = 374, 1238, 3
original_image_size = (img_rows, img_cols, img_chns)
latent_dim = 2
intermediate_dim = 512
epochs = 50
epsilon_std = 1.0
n_filters = 64

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
    input_img = Input(batch_shape=(batch_size,) + original_image_size)
    conv_1 = Conv2D(img_chns,
                    kernel_size=(3,3),
                    padding='same')(input_img)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)
    conv_1 = Dropout(0.2)(conv_1)

    conv_2 = Conv2D(n_filters,
                    kernel_size=(3,3),
                    padding='same',
                    strides=(1,1))(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)
    conv_2 = Dropout(0.2)(conv_2)

    conv_3 = Conv2D(n_filters,
                    kernel_size=(3, 3),
                    padding='same',
                    strides=(1, 1))(conv_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)
    conv_3 = Dropout(0.2)(conv_3)

    conv_4 = Conv2D(n_filters,
                    kernel_size=(3, 3),
                    padding='same',
                    strides=(1, 1))(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation('relu')(conv_4)
    conv_4 = Dropout(0.2)(conv_4)

    flat = Flatten()(conv_4)
    hidden = Dense(intermediate_dim, activation='relu')(flat)

    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_var])
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_hidden = Dense(intermediate_dim, activation='relu')
    decoder_upsample = Dense(n_filters * 14 * 14, activation='relu')




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
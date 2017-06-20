from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
np.random.seed(2 ** 10)
import os
import math
import sys
import config
from keras.datasets import cifar10
from keras.models import Model, Sequential
from keras.layers import Input, Activation, Dense, Flatten, Dropout, UpSampling2D, ConvLSTM2D, TimeDistributed
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import metrics
from keras import backend as K
K.set_image_dim_ordering('tf')

# if len(sys.argv) !=2:
#     dir_path = sys.argv[2]
#     print ('Usage: python <train.py> <data_dir>')

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
def create_model():
    print ("Creating model...")
    # input_img = Input(batch_shape=(batch_size,) + original_image_size)

    # print (original_image_size)
    model = Sequential()
    model.add(TimeDistributed(Conv2D(filters=128,
                                    input_shape=(None,) + config.original_image_size,
                                    kernel_size=(11,11),
                                    padding='valid',
                                    strides=4),
                              input_shape=(None,) + config.original_image_size))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('tanh')))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(TimeDistributed(Conv2D(filters=64,
                                    kernel_size=(5,5),
                                    padding='valid',
                                    strides=2)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('tanh')))
    model.add(TimeDistributed(Dropout(0.2)))

    # Add ConvLSTM layers here
    model.add(ConvLSTM2D(filters=64,
                            kernel_size=(3,3),
                            padding='same',
                            strides=1,
                            return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32,
                            kernel_size=(3,3),
                            padding='same',
                            strides=1,
                            return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=64,
                            kernel_size=(3, 3),
                            padding='same',
                            strides=1,
                            return_sequences=True))
    model.add(BatchNormalization())

    # Add Deconvolutional layers here
    model.add(TimeDistributed(Conv2DTranspose(filters=128,
                                               kernel_size=(5, 5),
                                               padding='valid',
                                               strides=(2, 2))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('tanh')))
    model.add(TimeDistributed(Dropout(0.2)))

    # upsamp_1 = UpSampling2D(size=(2,2))(conv_3)

    model.add(TimeDistributed(Conv2DTranspose(filters=1,
                                               kernel_size=(11, 11),
                                               padding='valid',
                                               strides=(4, 4))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('tanh')))
    model.add(TimeDistributed(Dropout(0.2)))

    # upsamp_2 = UpSampling2D(size=(4, 4))(conv_4)

    # reconstructed = Conv2D(filters=1,
    #                        kernel_size=(11, 11),
    #                        activation='tanh',
    #                        padding='same')(deconv_4)

    return model

# -------------------------------------------------
if __name__ == '__main__':
    model = create_model()
    model.compile(optimizer='adadelta', loss="binary_crossentropy", metrics=['accuracy'])

    # # LearningRateScheduler(schedule=schedule),
    # callbacks = [ModelCheckpoint(checkpoint_path + 'weights.{epoch:02d}-{val_acc:.2f}.hdf5',
    #                              monitor='val_acc',
    #                              verbose=1,
    #                              save_best_only=True,
    #                              mode='max'),
    #              TensorBoard(log_dir='/tmp/autoencoder')]
    #
    # print ("Saving model")
    # with open(os.path.join(model_path, 'autoencoder.json')) as f:
    #     f.write(model.to_json())

    if config.print_model_summary:
        print ("Model summary...")
        print (model.summary())

    # if save_model_plot:
    #     print ("Saving model plot...")
    #     from keras.utils.visualize_util import plot
    #     plot(model, to_file=os.path.join(model_path, 'autoencoder.png'), show_shapes=True)

    # if data_augmentation:
    #     # Data augmentation if corresponding bool parameter set true
    #     print ("Using real-time data augmentation")
    #
    #     train_datagen = ImageDataGenerator(
    #         featurewise_center=True,
    #         featurewise_std_normalization=True,
    #         zca_whitening=True,
    #         horizontal_flip=True)
    #
    #     train_datagen.fit(X_train)
    #
    #     print ("Running training...")
    #     # fit the model on the batches generated by train_datagen.flow()
    #     model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=False),
    #                         samples_per_epoch=X_train.shape[0],
    #                         nb_epoch=nb_epochs,
    #                         validation_data=train_datagen.flow(X_test, Y_test, batch_size=batch_size),
    #                         nb_val_samples=X_test.shape[0],
    #                         callbacks=callbacks)
    #
    # else:
    #     # Not using data augmentation
    #     model.fit(X_train, Y_train,
    #               batch_size=batch_size,
    #               nb_epoch=nb_epochs,
    #               validation_data=(X_test, Y_test),
    #               callbacks=callbacks,
    #               shuffle=False)
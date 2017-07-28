from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

np.random.seed(2 ** 10)
import os
from scripts import model_config as config
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten, Dropout, Lambda, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import metrics
from keras import backend as K
K.set_image_dim_ordering('tf')

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
    input_img = Input(batch_shape=(config.batch_size,) + config.original_image_size)
    conv_1 = Conv2D(config.img_chns,
                    kernel_size=(3,3),
                    padding='same')(input_img)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)
    conv_1 = Dropout(0.2)(conv_1)

    conv_2 = Conv2D(config.n_filters,
                    kernel_size=(3,3),
                    padding='same',
                    strides=(2,2))(conv_1)      # Works similar to max-pooling
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)
    conv_2 = Dropout(0.2)(conv_2)

    conv_3 = Conv2D(config.n_filters,
                    kernel_size=(3, 3),
                    padding='same',
                    strides=(1, 1))(conv_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)
    conv_3 = Dropout(0.2)(conv_3)

    conv_4 = Conv2D(config.n_filters,
                    kernel_size=(3, 3),
                    padding='same',
                    strides=(1, 1))(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation('relu')(conv_4)
    conv_4 = Dropout(0.2)(conv_4)

    flat = Flatten()(conv_4)
    hidden = Dense(config.intermediate_dim, activation='relu')(flat)

    z_mean = Dense(config.latent_dim)(hidden)
    z_log_var = Dense(config.latent_dim)(hidden)

    # Serves for the reparametrization trick z = mu + sigma*epsilon
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(config.batch_size, config.latent_dim), mean=0.,
                                  stddev=config.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_var])
    z = Lambda(sampling, output_shape=(config.latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_hidden = Dense(config.intermediate_dim, activation='relu')
    decoder_upsample = Dense(config.n_filters * 14 * 14, activation='relu')

    output_shape = (config.batch_size, 14, 14, config.n_filters)
    decoder_reshape = Reshape(output_shape[1:])

    decoder_deconv_1 = Conv2DTranspose(config.n_filters,
                                       kernel_size=config.n_conv,
                                       padding='same',
                                       strides=1)

    decoder_deconv_2 = Conv2DTranspose(config.n_filters,
                                       kernel_size=config.n_conv,
                                       padding='same',
                                       strides=1)

    output_shape = (config.batch_size, 29, 29, config.n_filters)
    decoder_deconv_3_upsamp = Conv2DTranspose(config.n_filters,
                                              kernel_size=(3,3),
                                              strides=(2,2),
                                              padding='valid',
                                              activation='relu')
    decoder_mean_squash = Conv2D(config.img_chns,
                                 kernel_size=2,
                                 padding='valid',
                                 activation='sigmoid')

    hidden_decoded = decoder_hidden(z)
    up_decoded = decoder_upsample(hidden_decoded)
    reshape_decoded = decoder_reshape(up_decoded)
    deconv_1_decoded = decoder_deconv_1(reshape_decoded)
    deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
    x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
    x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

    def vae_loss(x, x_decoded_mean):
        # NOTE: binary_crossentropy expects a config.batch_size by dim
        # for x and x_decoded_mean, so we MUST flatten these!
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = config.img_rows * config.img_cols * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    vae = Model(input_img, x_decoded_mean_squash)
    vae.compile(optimizer='rmsprop', loss=vae_loss)

    return model

# -------------------------------------------------
if __name__ == '__main__':
    model = create_model()
    model.compile(optimizer='adadelta', loss="categorical_crossentropy", metrics=['accuracy'])

    print (config.data_path)

    callbacks = [LearningRateScheduler(schedule=config.schedule),
                 ModelCheckpoint(config.checkpoint_path + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
                 ]

    print ("Saving model")
    with open(os.path.join(config.model_path, 'paintgan.json')) as f:
        f.write(model.to_json())

    if config.print_model_summary:
        print ("Model summary...")
        print (model.summary())

    if config.save_model_plot:
        print ("Saving model plot...")
        from keras.utils.visualize_util import plot
        plot(model, to_file=os.path.join(config.model_path, 'paintgan.png'), show_shapes=True)

    if config.data_augmentation:
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
        model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=config.batch_size, shuffle=True),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=config.nb_epochs,
                            validation_data=train_datagen.flow(X_test, Y_test, batch_size=config.batch_size),
                            nb_val_samples=X_test.shape[0],
                            callbacks=callbacks)

    else:
        # Not using data augmentation
        model.fit(X_train, Y_train,
                  batch_size=config.batch_size,
                  nb_epoch=config.nb_epochs,
                  validation_data=(X_test, Y_test),
                  callbacks=callbacks,
                  shuffle=True)
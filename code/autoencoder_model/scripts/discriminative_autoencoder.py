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
from keras.layers import Layer
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import Conv3DTranspose
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.merge import multiply
from keras.layers.merge import concatenate
from keras.layers.core import Permute
from keras.layers.core import RepeatVector
from keras.layers.core import Dense
from keras.layers.core import Lambda
from keras.layers.core import Reshape
from keras.layers.core import Flatten
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input
from keras.models import Model
from config_da import *

import tb_callback
import lrs_callback
import argparse
import math
import os
import cv2
from sys import stdout


def encoder_model():
    model = Sequential()

    # 10x128x128
    model.add(Conv3D(filters=128,
                     strides=(1, 4, 4),
                     kernel_size=(3, 11, 11),
                     padding='same',
                     input_shape=(int(VIDEO_LENGTH/2), 128, 128, 3)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
    model.add(TimeDistributed(Dropout(0.5)))

    # 10x32x32
    model.add(Conv3D(filters=64,
                     strides=(1, 2, 2),
                     kernel_size=(3, 5, 5),
                     padding='same'))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
    model.add(TimeDistributed(Dropout(0.5)))

    # 10x16x16
    model.add(Conv3D(filters=32,
                     strides=(1, 1, 1),
                     kernel_size=(3, 5, 5),
                     padding='same'))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
    model.add(TimeDistributed(Dropout(0.5)))

    return model


def decoder_model():
    inputs = Input(shape=(10, 16, 16, 32))

    # Learn alpha_1
    # flat_1 = TimeDistributed(Flatten())(inputs)
    # lstm_1 = LSTM(units=16*16,
    #               activation='softmax',
    #               return_sequences=True)(flat_1)
    # x = Flatten()(lstm_1)
    # x = RepeatVector(n=32)(x)
    # x = Permute((2, 1))(x)
    # x = Reshape(target_shape=(10, 16, 16, 32))(x)
    # attn_1 = multiply([inputs, x])

    # 10x64x64
    conv_1 = Conv3DTranspose(filters=64,
                             kernel_size=(3, 5, 5),
                             padding='same',
                             strides=(1, 1, 1))(inputs)
    x = TimeDistributed(BatchNormalization())(conv_1)
    x = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_1 = TimeDistributed(Dropout(0.5))(x)

    # 10x32x32
    conv_2 = Conv3DTranspose(filters=128,
                             kernel_size=(3, 5, 5),
                             padding='same',
                             strides=(1, 2, 2))(out_1)
    x = TimeDistributed(BatchNormalization())(conv_2)
    x = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_2 = TimeDistributed(Dropout(0.5))(x)

    # 10x64x64
    conv_3 = Conv3DTranspose(filters=64,
                             kernel_size=(3, 5, 5),
                             padding='same',
                             strides=(1, 2, 2))(out_2)
    x = TimeDistributed(BatchNormalization())(conv_3)
    x = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_3 = TimeDistributed(Dropout(0.5))(x)

    # Learn alpha_1
    conv3D_1 = Conv3D(filters=64,
                      strides=(1, 1, 1),
                      kernel_size=(3, 3, 3),
                      dilation_rate=(3, 3, 3),
                      padding='same')(out_3)
    x = TimeDistributed(BatchNormalization())(conv3D_1)
    x = TimeDistributed(Dropout(0.5))(x)

    conv3D_2 = Conv3D(filters=1,
                      strides=(1, 1, 1),
                      kernel_size=(3, 3, 3),
                      dilation_rate=(3, 3, 3),
                      padding='same')(x)
    x = TimeDistributed(BatchNormalization())(conv3D_2)
    x = TimeDistributed(Dropout(0.5))(x)
    flat_1 = TimeDistributed(Flatten())(x)

    # convlstm_1 = ConvLSTM2D(filters=1,
    #                         kernel_size=(3, 3),
    #                         strides=(1, 1),
    #                         dilation_rate=(2, 2),
    #                         padding='same',
    #                         return_sequences=True,
    #                         name='convlstm_1')(out_3)

    # convlstm_2 = ConvLSTM2D(filters=1,
    #                         kernel_size=(3, 3),
    #                         dilation_rate=(3, 3),
    #                         strides=(1, 1),
    #                         padding='same',
    #                         return_sequences=True,
    #                         name='convlstm_2')(convlstm_1)
    # flat_1 = TimeDistributed(Flatten())(convlstm_1)

    dense_1 = TimeDistributed(Dense(units=64*64, activation='softmax'))(flat_1)
    x = TimeDistributed(Dropout(0.5))(dense_1)
    x = Flatten()(x)
    x = RepeatVector(n=64)(x)
    x = Permute((2, 1))(x)
    x = Reshape(target_shape=(10, 64, 64, 64))(x)
    attn_1 = multiply([out_3, x])


    # 10x128x128
    conv_4 = Conv3DTranspose(filters=3,
                             kernel_size=(3, 11, 11),
                             strides=(1, 2, 2),
                             padding='same')(attn_1)
    x = TimeDistributed(BatchNormalization())(conv_4)
    x = TimeDistributed(Activation('tanh'))(x)
    predictions = TimeDistributed(Dropout(0.5))(x)

    model = Model(inputs=inputs ,outputs=predictions)

    return model


def discriminator_model():

    model = Sequential()
    model.add(Conv3D(filters=64,
                     kernel_size=(4, 4, 4),
                     padding='same',
                     input_shape=(int(VIDEO_LENGTH/2), 128, 128, 3)))
    # model.add(BatchNormalization())
    # model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
    model.add(TimeDistributed(Dropout(0.5)))

    model.add(Conv3D(filters=128,
                     kernel_size=(4, 4, 4),
                     strides=(2, 4, 4),
                     padding='same'))
    # model.add(BatchNormalization())
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
    model.add(TimeDistributed(Dropout(0.5)))

    model.add(Conv3D(filters=128,
                     kernel_size=(4, 4, 4),
                     strides=(2, 2, 2),
                     padding='same'))
    # model.add(BatchNormalization())
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
    model.add(TimeDistributed(Dropout(0.5)))

    model.add(Conv3D(filters=128,
                     kernel_size=(4, 4, 4),
                     strides=(2, 2, 2),
                     padding='same'))
    # model.add(BatchNormalization())
    model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
    model.add(TimeDistributed(Dropout(0.5)))

    model.add(Flatten())
    # model.add(Dense(256))
    # model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def set_trainability(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def autoencoder_model(encoder, decoder):
    model = Sequential()
    model.add(encoder)
    set_trainability(encoder, False)
    model.add(decoder)
    return model


def da_model(autoencoder, discriminator):
    model = Sequential()
    model.add(autoencoder)
    set_trainability(discriminator, False)
    model.add(discriminator)
    return model


def combine_images(generated_images, X, y):
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

    # Ground truth
    truth_frames = np.zeros((n_frames,) + y.shape[2:], dtype=y.dtype)
    frame_index = 0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            truth_frames[frame_index] = y[i, j]
            frame_index += 1

    num = truth_frames.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = truth_frames.shape[1:]
    truth_image = np.zeros((height * shape[0], width * shape[1], shape[2]), dtype=y.dtype)
    for index, img in enumerate(truth_frames):
        i = int(index / width)
        j = index % width
        truth_image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = img

    return orig_image, image, truth_image


def load_weights(weights_file, model):
    model.load_weights(weights_file)


def run_utilities(encoder, decoder, autoencoder, discriminator, ENC_WEIGHTS, DEC_WEIGHTS, DIS_WEIGHTS):
    if PRINT_MODEL_SUMMARY:
        print (encoder.summary())
        print (decoder.summary())
        print (autoencoder.summary())
        print (discriminator.summary())
        # exit(0)

    # Save model to file
    if SAVE_MODEL:
        print("Saving models to file...")
        model_json = encoder.to_json()
        with open(os.path.join(MODEL_DIR, "encoder.json"), "w") as json_file:
            json_file.write(model_json)

        model_json = decoder.to_json()
        with open(os.path.join(MODEL_DIR, "decoder.json"), "w") as json_file:
            json_file.write(model_json)

        model_json = autoencoder.to_json()
        with open(os.path.join(MODEL_DIR, "autoencoder.json"), "w") as json_file:
            json_file.write(model_json)

        model_json = discriminator.to_json()
        with open(os.path.join(MODEL_DIR, "discriminator.json"), "w") as json_file:
            json_file.write(model_json)

        if PLOT_MODEL:
            plot_model(encoder, to_file=os.path.join(MODEL_DIR, 'encoder.png'), show_shapes=True)
            plot_model(decoder, to_file=os.path.join(MODEL_DIR, 'decoder.png'), show_shapes=True)
            plot_model(autoencoder, to_file=os.path.join(MODEL_DIR, 'autoencoder.png'), show_shapes=True)
            plot_model(discriminator, to_file=os.path.join(MODEL_DIR, 'discriminator.png'), show_shapes=True)

    if ENC_WEIGHTS != "None":
        print ("Pre-loading encoder with weights...")
        load_weights(ENC_WEIGHTS, encoder)
    if DEC_WEIGHTS != "None":
        print ("Pre-loading decoder with weights...")
        load_weights(DEC_WEIGHTS, decoder)
    if DIS_WEIGHTS != "None":
        print("Pre-loading decoder with weights...")
        load_weights(DIS_WEIGHTS, discriminator)


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
                print (e)

    return X


def train(BATCH_SIZE, ENC_WEIGHTS, DEC_WEIGHTS, DIS_WEIGHTS):
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

    # Build the Spatio-temporal Autoencoder
    print ("Creating models...")
    encoder = encoder_model()
    decoder = decoder_model()
    discriminator = discriminator_model()
    autoencoder = autoencoder_model(encoder, decoder)
    da = da_model(autoencoder, discriminator)

    run_utilities(encoder, decoder, autoencoder, discriminator, ENC_WEIGHTS, DEC_WEIGHTS, DIS_WEIGHTS)

    da.compile(loss='binary_crossentropy', optimizer=OPTIM_A)
    set_trainability(discriminator, True)
    discriminator.compile(loss='binary_crossentropy', optimizer=OPTIM_D)
    NB_ITERATIONS = int(n_videos/BATCH_SIZE)

    # Setup TensorBoard Callback
    TC = tb_callback.TensorBoard(log_dir=TF_LOG_DIR, histogram_freq=0, write_graph=False, write_images=False)
    # LRS = lrs_callback.LearningRateScheduler(schedule=schedule)
    # LRS.set_model(autoencoder)

    print ("Beginning Training...")
    # Begin Training
    for epoch in range(NB_EPOCHS):
        print("\n\nEpoch ", epoch)
        a_loss = []
        d_loss = []
        # Set learning rate every epoch
        # LRS.on_epoch_begin(epoch=epoch)
        # lr = K.get_value(autoencoder.optimizer.lr)
        # print ("Learning rate: " + str(lr))

        for index in range(NB_ITERATIONS):
            # Train Autoencoder
            X = load_X(videos_list, index, DATA_DIR)
            X_train = X[:, 0 : int(VIDEO_LENGTH/2)]
            y_train = X[:, int(VIDEO_LENGTH/2) :]

            # Adversarially training a pre-trained autoencoder
            future_images = autoencoder.predict(X_train, verbose=0)

            # Train Discriminator on future images (y_train, not X_train)
            X = np.concatenate((y_train, future_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss.append(discriminator.train_on_batch(X, y))

            # Train Discriminative Autoencoder
            set_trainability(discriminator, False)
            a_loss.append(da.train_on_batch(X_train, [1] * BATCH_SIZE))
            set_trainability(discriminator, True)

            arrow = int(index / (NB_ITERATIONS / 40))
            stdout.write("\rIteration: " + str(index) + "/" + str(NB_ITERATIONS - 1) + "  " +
                         "a_loss: " + str(a_loss[len(a_loss) - 1]) + "\t    " + "d_loss: " +
                         str(d_loss[len(d_loss) - 1]) + "\t    [" + "{0}>".format("=" * (arrow)))
            stdout.flush()

        if SAVE_GENERATED_IMAGES:
            # Save generated images to file
            generated_images = autoencoder.predict(X_train, verbose=0)
            orig_image, image, truth_image = combine_images(generated_images, X_train, y_train)
            image = image * 127.5 + 127.5
            orig_image = orig_image * 127.5 + 127.5
            truth_image = truth_image * 127.5 + 127.5
            if epoch == 0 :
                cv2.imwrite(os.path.join(GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_orig.png"), orig_image)
                cv2.imwrite(os.path.join(GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_truth.png"), truth_image)
            cv2.imwrite(os.path.join(GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + ".png"), image)

        # then after each epoch/iteration
        avg_a_loss = sum(a_loss)/len(a_loss)
        avg_d_loss = sum(d_loss) / len(d_loss)
        logs = {'a_loss': avg_a_loss, 'd_loss': avg_d_loss}
        TC.on_epoch_end(epoch, logs)

        # Log the losses
        with open(os.path.join(LOG_DIR, 'losses.json'), 'a') as log_file:
            log_file.write("{\"epoch\":%d, \"a_loss\":%f, \"d_loss\":%f};\n" % (epoch, avg_a_loss, avg_d_loss))

        print("\nAvg a_loss: " + str(avg_a_loss) + "  Avg d_loss: " + str(avg_d_loss))

        # Save model weights per epoch to file
        encoder.save_weights(os.path.join(CHECKPOINT_DIR, 'encoder_epoch_'+str(epoch)+'.h5'), True)
        decoder.save_weights(os.path.join(CHECKPOINT_DIR, 'decoder_epoch_' + str(epoch) + '.h5'), True)
        discriminator.save_weights(os.path.join(CHECKPOINT_DIR, 'discriminator_epoch_' + str(epoch) + '.h5'), True)

    # End TensorBoard Callback
    TC.on_train_end('_')


def test(ENC_WEIGHTS, DEC_WEIGHTS, DIS_WEIGHTS):

    # Create models
    print ("Creating models...")
    encoder = encoder_model()
    decoder = decoder_model()
    discriminator = discriminator_model()
    autoencoder = autoencoder_model(encoder, decoder)
    da = da_model(autoencoder, discriminator)

    run_utilities(encoder, decoder, autoencoder, discriminator, ENC_WEIGHTS, DEC_WEIGHTS, DIS_WEIGHTS)
    da.compile(loss='binary_crossentropy', optimizer=OPTIM_A)

    for i in range(len(decoder.layers)):
        print (decoder.layers[i], str(i))

    exit(0)

    def build_intermediate_model(encoder, decoder):
        intermediate_decoder_1 = Model(inputs=decoder.layers[0].input, outputs=decoder.layers[18].output)
        # intermediate_decoder_2 = Model(inputs=decoder.layers[0].input, outputs=decoder.layers[12].output)

        imodel_1 = Sequential()
        imodel_1.add(encoder)
        imodel_1.add(intermediate_decoder_1)

        # imodel_2 = Sequential()
        # imodel_2.add(encoder)
        # imodel_2.add(intermediate_decoder_2)

        return imodel_1

    imodel_1 = build_intermediate_model(encoder, decoder)
    imodel_1.compile(loss='mean_squared_error', optimizer=OPTIM)
    # imodel_2.compile(loss='mean_squared_error', optimizer=OPTIM)

    # imodel = build_intermediate_model(encoder, decoder)

    # Build video progressions
    frames_source = hkl.load(os.path.join(TEST_DATA_DIR, 'sources_test_128.hkl'))
    videos_list = []
    start_frame_index = 1
    end_frame_index = VIDEO_LENGTH + 1
    while (end_frame_index <= len(frames_source)):
        frame_list = frames_source[start_frame_index:end_frame_index]
        if (len(set(frame_list)) == 1):
            videos_list.append(range(start_frame_index, end_frame_index))
            start_frame_index = start_frame_index + VIDEO_LENGTH
            end_frame_index = end_frame_index + VIDEO_LENGTH
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
        X = load_X(videos_list, index, TEST_DATA_DIR)
        X_test = X[:, 0: int(VIDEO_LENGTH / 2)]
        y_test = X[:, int(VIDEO_LENGTH / 2):]
        loss.append(autoencoder.test_on_batch(X_test, y_test))
        y_pred = autoencoder.predict_on_batch(X_test)
        a_pred_1 = imodel_1.predict_on_batch(X_test)
        # a_pred_2 = imodel_2.predict_on_batch(X_test)

        arrow = int(index / (NB_ITERATIONS / 40))
        stdout.write("\rIteration: " + str(index) + "/" + str(NB_ITERATIONS - 1) + "  " +
                     "loss: " + str(loss[len(loss) - 1]) +
                     "\t    [" + "{0}>".format("=" * (arrow)))
        stdout.flush()

        orig_image, truth_image, pred_image = combine_images(X_test, y_test, y_pred)
        pred_image = pred_image * 127.5 + 127.5
        orig_image = orig_image * 127.5 + 127.5
        truth_image = truth_image * 127.5 + 127.5

        cv2.imwrite(os.path.join(TEST_RESULTS_DIR, str(index) + "_orig.png"), orig_image)
        cv2.imwrite(os.path.join(TEST_RESULTS_DIR, str(index) + "_truth.png"), truth_image)
        cv2.imwrite(os.path.join(TEST_RESULTS_DIR, str(index) + "_pred.png"), pred_image)

        #------------------------------------------
        a_pred_1 = np.reshape(a_pred_1, newshape=(10, 10, 64, 64, 1))
        np.save(os.path.join(TEST_RESULTS_DIR, 'attention_weights_' + str(index) +'.npy'), a_pred_1)
        orig_image, truth_image, pred_image = combine_images(X_test, y_test, a_pred_1)
        pred_image = (pred_image*100) * 127.5 + 127.5
        y_pred = y_pred * 127.5 + 127.5
        # np.save(os.path.join(TEST_RESULTS_DIR, 'attention_weights_' + str(index) + '.npy'), y_pred)
        cv2.imwrite(os.path.join(TEST_RESULTS_DIR, str(index) + "_attn_1.png"), pred_image)

        # a_pred_2 = np.reshape(a_pred_2, newshape=(10, 10, 16, 16, 1))
        # with open('attention_weights.txt', mode='w') as file:
        #     file.write(str(a_pred_2[0, 4]))
        # orig_image, truth_image, pred_image = combine_images(X_test, y_test, a_pred_2)
        # pred_image = (pred_image*100) * 127.5 + 127.5
        # cv2.imwrite(os.path.join(TEST_RESULTS_DIR, str(index) + "_attn_2.png"), pred_image)

    avg_loss = sum(loss) / len(loss)
    print("\nAvg loss: " + str(avg_loss))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--enc_weights", type=str, default="None")
    parser.add_argument("--dec_weights", type=str, default="None")
    parser.add_argument("--dis_weights", type=str, default="None")
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
              DIS_WEIGHTS=args.dis_weights)

    if args.mode == "test":
        test(ENC_WEIGHTS=args.enc_weights,
             DEC_WEIGHTS=args.dec_weights,
             DIS_WEIGHTS=args.dis_weights)
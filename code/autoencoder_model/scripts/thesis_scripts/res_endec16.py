from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hickle as hkl
import numpy as np

np.random.seed(9 ** 10)
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras import regularizers
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers.core import Activation
from keras.utils.vis_utils import plot_model
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import UpSampling3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from plot_results import plot_err_variation
from keras.layers import Input
from keras.models import Model
from config_rendec16 import *
from sys import stdout
from keras.layers.core import Lambda

import tb_callback
import lrs_callback
import argparse
import cv2
import os


def encoder_model():
    inputs = Input(shape=(int(VIDEO_LENGTH/2), 128, 208, 3))

    # 10x128x128
    conv_1 = Conv3D(filters=128,
                     strides=(1, 4, 4),
                     dilation_rate=(1, 1, 1),
                     kernel_size=(3, 11, 11),
                     padding='same')(inputs)
    x = TimeDistributed(BatchNormalization())(conv_1)
    x = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_1 = TimeDistributed(Dropout(0.5))(x)

    conv_2a = Conv3D(filters=64,
                     strides=(1, 1, 1),
                     dilation_rate=(2, 1, 1),
                     kernel_size=(2, 5, 5),
                     padding='same')(out_1)
    x = TimeDistributed(BatchNormalization())(conv_2a)
    x = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_2a = TimeDistributed(Dropout(0.5))(x)

    conv_2b = Conv3D(filters=64,
                    strides=(1, 1, 1),
                    dilation_rate=(2, 1, 1),
                    kernel_size=(2, 5, 5),
                    padding='same')(out_2a)
    x = TimeDistributed(BatchNormalization())(conv_2b)
    x = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_2b = TimeDistributed(Dropout(0.5))(x)

    conv_2c = TimeDistributed(Conv2D(filters=64,
                                        kernel_size=(1, 1),
                                        strides=(1, 1),
                                        padding='same'))(out_1)
    x = TimeDistributed(BatchNormalization())(conv_2c)
    out_1_less = TimeDistributed(LeakyReLU(alpha=0.2))(x)

    res_1 = add([out_1_less, out_2b])
    # res_1 = LeakyReLU(alpha=0.2)(res_1)

    conv_3 = Conv3D(filters=64,
                     strides=(1, 2, 2),
                     dilation_rate=(1, 1, 1),
                     kernel_size=(3, 5, 5),
                     padding='same')(res_1)
    x = TimeDistributed(BatchNormalization())(conv_3)
    x = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_3 = TimeDistributed(Dropout(0.5))(x)

    # 10x16x16
    conv_4a = Conv3D(filters=64,
                     strides=(1, 1, 1),
                     dilation_rate=(2, 1, 1),
                     kernel_size=(2, 3, 3),
                     padding='same')(out_3)
    x = TimeDistributed(BatchNormalization())(conv_4a)
    x = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_4a = TimeDistributed(Dropout(0.5))(x)

    conv_4b = Conv3D(filters=64,
                     strides=(1, 1, 1),
                     dilation_rate=(2, 1, 1),
                     kernel_size=(2, 3, 3),
                     padding='same')(out_4a)
    x = TimeDistributed(BatchNormalization())(conv_4b)
    x = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    out_4b = TimeDistributed(Dropout(0.5))(x)

    z = add([out_3, out_4b])
    # res_1 = LeakyReLU(alpha=0.2)(res_1)

    model = Model(inputs=inputs, outputs=[z, res_1])

    return model


def decoder_model():
    inputs = Input(shape=(int(VIDEO_LENGTH/2), 16, 26, 64))
    residual_input = Input(shape=(int(VIDEO_LENGTH/2), 32, 52, 64), name='res_input')

    # Adjust residual input
    def adjust_res(x):
        pad = K.zeros_like(x[:, 1:])
        res = x[:, 0:1]
        return K.concatenate([res, pad], axis=1)

    enc_input = Lambda(adjust_res)(residual_input)

    # 10x16x16
    convlstm_1 = ConvLSTM2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(inputs)
    x = TimeDistributed(BatchNormalization())(convlstm_1)
    out_1 = TimeDistributed(Activation('tanh'))(x)

    convlstm_2 = ConvLSTM2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(out_1)
    x = TimeDistributed(BatchNormalization())(convlstm_2)
    out_2 = TimeDistributed(Activation('tanh'))(x)

    res_1 = add([inputs, out_2])
    res_1 = UpSampling3D(size=(1, 2, 2))(res_1)

    # 10x32x32
    convlstm_3a = ConvLSTM2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(res_1)
    x = TimeDistributed(BatchNormalization())(convlstm_3a)
    out_3a = TimeDistributed(Activation('tanh'))(x)

    convlstm_3b = ConvLSTM2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(out_3a)
    x = TimeDistributed(BatchNormalization())(convlstm_3b)
    out_3b = TimeDistributed(Activation('tanh'))(x)

    res_2 = add([res_1, out_3b, enc_input])
    res_2 = UpSampling3D(size=(1, 2, 2))(res_2)

    # 10x64x64
    convlstm_4a = ConvLSTM2D(filters=16,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(res_2)
    x = TimeDistributed(BatchNormalization())(convlstm_4a)
    out_4a = TimeDistributed(Activation('tanh'))(x)

    convlstm_4b = ConvLSTM2D(filters=16,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(out_4a)
    x = TimeDistributed(BatchNormalization())(convlstm_4b)
    out_4b = TimeDistributed(Activation('tanh'))(x)

    conv_4c = TimeDistributed(Conv2D(filters=16,
                                        kernel_size=(1, 1),
                                        strides=(1, 1),
                                        padding='same'))(res_2)
    x = TimeDistributed(BatchNormalization())(conv_4c)
    res_2_less = TimeDistributed(Activation('tanh'))(x)
    res_3 = add([res_2_less, out_4b])
    res_3 = UpSampling3D(size=(1, 2, 2))(res_3)

    # 10x128x128
    convlstm_5 = ConvLSTM2D(filters=3,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(res_3)
    predictions = TimeDistributed(Activation('tanh'))(convlstm_5)

    model = Model(inputs=[inputs, residual_input], outputs=predictions)

    return model


def set_trainability(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def autoencoder_model(encoder, decoder):
    # model = Sequential()
    # model.add(encoder)
    # model.add(decoder)

    inputs = Input(shape=(int(VIDEO_LENGTH / 2), 128, 208, 3))
    z, res = encoder(inputs)
    future = decoder([z, res])

    model = Model(inputs=inputs, outputs=future)

    return model


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


def run_utilities(encoder, decoder, autoencoder, ENC_WEIGHTS, DEC_WEIGHTS):
    if PRINT_MODEL_SUMMARY:
        print(encoder.summary())
        print(decoder.summary())
        print(autoencoder.summary())
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

        if PLOT_MODEL:
            plot_model(encoder, to_file=os.path.join(MODEL_DIR, 'encoder.png'), show_shapes=True)
            plot_model(decoder, to_file=os.path.join(MODEL_DIR, 'decoder.png'), show_shapes=True)
            plot_model(autoencoder, to_file=os.path.join(MODEL_DIR, 'autoencoder.png'), show_shapes=True)

    if ENC_WEIGHTS != "None":
        print("Pre-loading encoder with weights...")
        load_weights(ENC_WEIGHTS, encoder)
    if DEC_WEIGHTS != "None":
        print("Pre-loading decoder with weights...")
        load_weights(DEC_WEIGHTS, decoder)


def load_to_RAM(frames_source):
    frames = np.zeros(shape=((len(frames_source),) + IMG_SIZE))
    print("Decimating RAM!")
    j = 1
    for i in range(1, len(frames_source)):
        filename = "frame_" + str(j) + ".png"
        im_file = os.path.join(DATA_DIR, filename)
        try:
            frame = cv2.imread(im_file, cv2.IMREAD_COLOR)
            frames[i] = (frame.astype(np.float32) - 127.5) / 127.5
            j = j + 1
        except AttributeError as e:
            print(im_file)
            print(e)

    return frames


def load_X_RAM(videos_list, index, frames):
    X = []
    for i in range(BATCH_SIZE):
        start_index = videos_list[(index * BATCH_SIZE + i), 0]
        end_index = videos_list[(index * BATCH_SIZE + i), -1]
        X.append(frames[start_index:end_index + 1])
    X = np.asarray(X)

    return X


def load_X(videos_list, index, data_dir, img_size, batch_size=BATCH_SIZE):
    X = np.zeros((batch_size, VIDEO_LENGTH,) + img_size)
    for i in range(batch_size):
        for j in range(VIDEO_LENGTH):
            filename = "frame_" + str(videos_list[(index * batch_size + i), j]) + ".png"
            im_file = os.path.join(data_dir, filename)
            try:
                frame = cv2.imread(im_file, cv2.IMREAD_COLOR)
                X[i, j] = (frame.astype(np.float32) - 127.5) / 127.5
            except AttributeError as e:
                print(im_file)
                print(e)

    return X


def get_video_lists(frames_source, stride):
    # Build video progressions
    videos_list = []
    start_frame_index = 1
    end_frame_index = VIDEO_LENGTH + 1
    while (end_frame_index <= len(frames_source)):
        frame_list = frames_source[start_frame_index:end_frame_index]
        if (len(set(frame_list)) == 1):
            videos_list.append(range(start_frame_index, end_frame_index))
            start_frame_index = start_frame_index + stride
            end_frame_index = end_frame_index + stride
        else:
            start_frame_index = end_frame_index - 1
            end_frame_index = start_frame_index + VIDEO_LENGTH

    videos_list = np.asarray(videos_list, dtype=np.int32)

    return np.asarray(videos_list)


def train(BATCH_SIZE, ENC_WEIGHTS, DEC_WEIGHTS):
    print("Loading data definitions...")
    frames_source = hkl.load(os.path.join(DATA_DIR, 'sources_train_208.hkl'))
    videos_list = get_video_lists(frames_source=frames_source, stride=4)
    n_videos = videos_list.shape[0]

    # Setup test
    val_frames_source = hkl.load(os.path.join(VAL_DATA_DIR, 'sources_val_208.hkl'))
    val_videos_list = get_video_lists(frames_source=val_frames_source, stride=(int(VIDEO_LENGTH / 2)))
    n_val_videos = val_videos_list.shape[0]

    if RAM_DECIMATE:
        frames = load_to_RAM(frames_source=frames_source)

    if SHUFFLE:
        # Shuffle images to aid generalization
        videos_list = np.random.permutation(videos_list)

    # Build the Spatio-temporal Autoencoder
    print("Creating models...")
    encoder = encoder_model()
    decoder = decoder_model()
    autoencoder = autoencoder_model(encoder, decoder)
    autoencoder.compile(loss="mean_squared_error", optimizer=OPTIM_A)

    run_utilities(encoder, decoder, autoencoder, ENC_WEIGHTS, DEC_WEIGHTS)

    NB_ITERATIONS = int(n_videos / BATCH_SIZE)
    # NB_ITERATIONS = 5
    NB_VAL_ITERATIONS = int(n_val_videos / BATCH_SIZE)

    # Setup TensorBoard Callback
    TC = tb_callback.TensorBoard(log_dir=TF_LOG_DIR, histogram_freq=0, write_graph=False, write_images=False)
    LRS = lrs_callback.LearningRateScheduler(schedule=schedule)
    LRS.set_model(autoencoder)

    print("Beginning Training...")
    # Begin Training
    for epoch in range(1, NB_EPOCHS_AUTOENCODER+1):
        if epoch == 21:
            autoencoder.compile(loss="mean_absolute_error", optimizer=OPTIM_B)
            load_weights(os.path.join(CHECKPOINT_DIR, 'encoder_epoch_20.h5'), encoder)
            load_weights(os.path.join(CHECKPOINT_DIR, 'decoder_epoch_20.h5'), decoder)

        print("\n\nEpoch ", epoch)
        loss = []
        val_loss = []

        # Set learning rate every epoch
        LRS.on_epoch_begin(epoch=epoch)
        lr = K.get_value(autoencoder.optimizer.lr)
        print("Learning rate: " + str(lr))

        for index in range(NB_ITERATIONS):
            # Train Autoencoder
            if RAM_DECIMATE:
                X = load_X_RAM(videos_list, index, frames)
            else:
                X = load_X(videos_list, index, DATA_DIR, IMG_SIZE)
            X_train = np.flip(X[:, 0: int(VIDEO_LENGTH / 2)], axis=1)
            y_train = X[:, int(VIDEO_LENGTH / 2):]
            loss.append(autoencoder.train_on_batch(X_train, y_train))

            arrow = int(index / (NB_ITERATIONS / 40))
            stdout.write("\rIter: " + str(index) + "/" + str(NB_ITERATIONS - 1) + "  " +
                         "loss: " + str(loss[len(loss) - 1]) +
                         "\t    [" + "{0}>".format("=" * (arrow)))
            stdout.flush()

        if SAVE_GENERATED_IMAGES:
            # Save generated images to file
            predicted_images = autoencoder.predict(X_train, verbose=0)
            voila = np.concatenate((X_train, y_train), axis=1)
            truth_seq = arrange_images(voila)
            pred_seq = arrange_images(np.concatenate((X_train, predicted_images), axis=1))

            truth_seq = truth_seq * 127.5 + 127.5
            pred_seq = pred_seq * 127.5 + 127.5

            if epoch == 1:
                cv2.imwrite(os.path.join(GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_truth.png"), truth_seq)
            cv2.imwrite(os.path.join(GEN_IMAGES_DIR, str(epoch) + "_" + str(index) + "_pred.png"), pred_seq)

        # Run over test data
        print('')
        for index in range(NB_VAL_ITERATIONS):
            X = load_X(val_videos_list, index, VAL_DATA_DIR, IMG_SIZE)
            X_val = np.flip(X[:, 0: int(VIDEO_LENGTH / 2)], axis=1)
            y_val = X[:, int(VIDEO_LENGTH / 2):]
            val_loss.append(autoencoder.test_on_batch(X_val, y_val))

            arrow = int(index / (NB_VAL_ITERATIONS / 40))
            stdout.write("\rIter: " + str(index) + "/" + str(NB_VAL_ITERATIONS - 1) + "  " +
                         "val_loss: " + str(val_loss[len(val_loss) - 1]) +
                         "\t    [" + "{0}>".format("=" * (arrow)))
            stdout.flush()

        # then after each epoch/iteration
        avg_loss = sum(loss) / len(loss)
        avg_val_loss = sum(val_loss) / len(val_loss)
        logs = {'loss': avg_loss, 'val_loss': avg_val_loss}
        TC.on_epoch_end(epoch, logs)

        # Log the losses
        with open(os.path.join(LOG_DIR, 'losses_gen.json'), 'a') as log_file:
            log_file.write("{\"epoch\":%d, \"train_loss\":%f, \"val_loss\":%f}\n" % (epoch, avg_loss, avg_val_loss))

            print("\nAvg train loss: " + str(avg_loss) + " Avg val loss: " + str(avg_val_loss))

        # Save model weights per epoch to file
        if epoch > 15 and epoch < 21:
            encoder.save_weights(os.path.join(CHECKPOINT_DIR, 'encoder_epoch_' + str(epoch) + '.h5'), True)
            decoder.save_weights(os.path.join(CHECKPOINT_DIR, 'decoder_epoch_' + str(epoch) + '.h5'), True)
        if epoch > 25:
            encoder.save_weights(os.path.join(CHECKPOINT_DIR, 'encoder_epoch_' + str(epoch) + '.h5'), True)
            decoder.save_weights(os.path.join(CHECKPOINT_DIR, 'decoder_epoch_' + str(epoch) + '.h5'), True)

        # encoder.save_weights(os.path.join(CHECKPOINT_DIR, 'encoder_epoch_' + str(epoch) + '.h5'), True)
        # decoder.save_weights(os.path.join(CHECKPOINT_DIR, 'decoder_epoch_' + str(epoch) + '.h5'), True)

        # test(os.path.join(CHECKPOINT_DIR, 'encoder_epoch_' + str(epoch) + '.h5'),
        #      os.path.join(CHECKPOINT_DIR, 'decoder_epoch_' + str(epoch) + '.h5'))


def test(ENC_WEIGHTS, DEC_WEIGHTS):
    print('')
    # Setup test
    test_frames_source = hkl.load(os.path.join(TEST_DATA_DIR, 'sources_test_208.hkl'))
    test_videos_list = get_video_lists(frames_source=test_frames_source, stride=(int(VIDEO_LENGTH / 2)))
    n_test_videos = test_videos_list.shape[0]

    if not os.path.exists(TEST_RESULTS_DIR + '/truth/'):
        os.mkdir(TEST_RESULTS_DIR + '/truth/')
    if not os.path.exists(TEST_RESULTS_DIR + '/pred/'):
        os.mkdir(TEST_RESULTS_DIR + '/pred/')
    if not os.path.exists(TEST_RESULTS_DIR + '/graphs/'):
        os.mkdir(TEST_RESULTS_DIR + '/graphs/')
        os.mkdir(TEST_RESULTS_DIR + '/graphs/values/')

    print("Creating models...")
    encoder = encoder_model()
    decoder = decoder_model()
    autoencoder = autoencoder_model(encoder, decoder)
    autoencoder.compile(loss="mean_absolute_error", optimizer=OPTIM_A)

    run_utilities(encoder, decoder, autoencoder, ENC_WEIGHTS, DEC_WEIGHTS)

    NB_TEST_ITERATIONS = int(n_test_videos / TEST_BATCH_SIZE)
    test_loss = []
    mae_errors = np.zeros(shape=(n_test_videos, int(VIDEO_LENGTH/2) + 1))
    mse_errors = np.zeros(shape=(n_test_videos, int(VIDEO_LENGTH/2) + 1))

    z_all = []
    for index in range(NB_TEST_ITERATIONS):
        X = load_X(test_videos_list, index, TEST_DATA_DIR, IMG_SIZE, batch_size=TEST_BATCH_SIZE)
        X_test = np.flip(X[:, 0: int(VIDEO_LENGTH / 2)], axis=1)
        y_test = X[:, int(VIDEO_LENGTH / 2):]
        test_loss.append(autoencoder.test_on_batch(X_test, y_test))

        arrow = int(index / (NB_TEST_ITERATIONS / 40))
        stdout.write("\rIter: " + str(index) + "/" + str(NB_TEST_ITERATIONS - 1) + "  " +
                     "test_loss: " + str(test_loss[len(test_loss) - 1]) +
                     "\t    [" + "{0}>".format("=" * (arrow)))
        stdout.flush()


        if SAVE_GENERATED_IMAGES:
            # Save generated images to file
            z, res = encoder.predict(X_test, verbose=0)
            z_all.append(z)

            # z_new = np.zeros(shape=(TEST_BATCH_SIZE, 1, 16, 26, 64))
            # z_new[0] = z[:, 15]
            # z_new = np.repeat(z_new, int(VIDEO_LENGTH/2), axis=1)
            # predicted_images = decoder.predict(z, verbose=0)
            # voila = np.concatenate((X_test, y_test), axis=1)
            # truth_seq = arrange_images(voila)
            # pred_seq = arrange_images(np.concatenate((X_test, predicted_images), axis=1))
            #
            # truth_seq = truth_seq * 127.5 + 127.5
            # pred_seq = pred_seq * 127.5 + 127.5
            #
            # mae_error = []
            # mse_error = []
            # for i in range(int(VIDEO_LENGTH / 2)):
            #     mae_errors[index, i] = (mae(y_test[0, i].flatten(), predicted_images[0, i].flatten()))
            #     mae_error.append(mae_errors[index, i])
            #
            #
            #     mse_errors[index, i] = (mse(y_test[0, i].flatten(), predicted_images[0, i].flatten()))
            #     mse_error.append(mse_errors[index, i])
            #
            # dc_mae = mae(X_test[0, 0].flatten(), predicted_images[0, 0].flatten())
            # mae_errors[index, -1] = dc_mae
            # dc_mse = mse(X_test[0, 0].flatten(), predicted_images[0, 0].flatten())
            # mse_errors[index, -1] = dc_mse
            # cv2.imwrite(os.path.join(TEST_RESULTS_DIR + '/truth/', str(index) + "_truth.png"), truth_seq)
            # cv2.imwrite(os.path.join(TEST_RESULTS_DIR + '/pred/', str(index) + "_pred.png"), pred_seq)
            # plot_err_variation(mae_error, index, dc_mae, 'mae')
            # plot_err_variation(mse_error, index, dc_mse, 'mse')

    # np.save(os.path.join(TEST_RESULTS_DIR + '/graphs/values/', str(index) + "_mae.npy"), np.asarray(mae_errors))
    # np.save(os.path.join(TEST_RESULTS_DIR + '/graphs/values/', str(index) + "_mse.npy"), np.asarray(mse_errors))
    np.save(os.path.join(TEST_RESULTS_DIR + '/graphs/values/', "z_all.npy"), np.asarray(z_all))

    # then after each epoch/iteration
    avg_test_loss = sum(test_loss) / len(test_loss)
    np.save(TEST_RESULTS_DIR + 'test_loss.npy', np.asarray(test_loss))
    print("\nAvg loss: " + str(avg_test_loss))
    print("\n Std: " + str(np.std(np.asarray(test_loss))))
    print("\n Variance: " + str(np.var(np.asarray(test_loss))))
    print("\n Mean: " + str(np.mean(np.asarray(test_loss))))
    print("\n Max: " + str(np.max(np.asarray(test_loss))))
    print("\n Min: " + str(np.min(np.asarray(test_loss))))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--enc_weights", type=str, default="None")
    parser.add_argument("--dec_weights", type=str, default="None")
    parser.add_argument("--gen_weights", type=str, default="None")
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
              DEC_WEIGHTS=args.dec_weights)

    if args.mode == "test":
        test(ENC_WEIGHTS=args.enc_weights,
             DEC_WEIGHTS=args.dec_weights)

    # if args.mode == "test_ind":
    #     test_ind(ENC_WEIGHTS=args.enc_weights,
    #              DEC_WEIGHTS=args.dec_weights)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Layer
from keras import backend as K
K.set_image_dim_ordering('tf')
from config_ac import KL_COEFF, ATTN_COEFF

# Custom loss layer
class AttnLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(AttnLossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(AttnLossLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def attn_loss(self, a):
        attn_loss = K.mean(K.mean(K.square(1 - K.sum(a, axis=1)), axis=1), axis=1)
        return ATTN_COEFF * K.mean(attn_loss)

    def call(self, inputs):
        x = inputs
        loss = self.attn_loss(x)
        self.add_loss(loss, inputs=inputs)
        # We do use this output.
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

def broadcast_channels(x):
    return K.repeat_elements(x, 128, axis=-1)

def broadcast_output_shape(input_shape):
    return input_shape[0:3] + (128,)

def expectation(x):
    return K.sum(K.sum(x, axis=-2), axis=-2)

def mse_kld_loss(y_pred, y_true):
    mse_loss = K.mean(K.square(y_pred - y_true), axis=-1)

    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    kld_loss = K.sum(y_true * K.log(y_true / y_pred), axis=-1)

    return mse_loss + (KL_COEFF*kld_loss)

# def expectation_output_shape(input_shape):
#     return input_shape[0:3] + (128,)

# aclstm_1 = ConvLSTM2D(filters=1,
    #                       kernel_size=(3, 3),
    #                       dilation_rate=(2, 2),
    #                       strides=(1, 1),
    #                       padding='same',
    #                       return_sequences=True,
    #                       recurrent_dropout=0.5,
    #                       name='aclstm_1')(out_4)
    # x = TimeDistributed(BatchNormalization())(aclstm_1)
    # flat_1 = TimeDistributed(Flatten())(x)
    # dense_1 = TimeDistributed(Dense(units=64 * 64, activation='softmax',
    #                                 kernel_initializer=RandomNormal(mean=0.5, stddev=0.125)))(flat_1)
    # x = TimeDistributed(Dropout(0.5))(dense_1)
    # a_1 = Reshape(target_shape=(10, 64, 64, 1))(x)
    #
    # # aclstm_2 = ConvLSTM2D(filters=1,
    # #                       kernel_size=(3, 3),
    # #                       dilation_rate=(2, 2),
    # #                       strides=(1, 1),
    # #                       padding='same',
    # #                       return_sequences=True,
    # #                       recurrent_dropout=0.5,
    # #                       name='aclstm_2')(a_1)
    # # x = TimeDistributed(BatchNormalization())(aclstm_2)
    # # flat_2 = TimeDistributed(Flatten())(x)
    # # dense_2 = TimeDistributed(Dense(units=64 * 64, activation='softmax'))(flat_2)
    # # x = TimeDistributed(Dropout(0.5))(dense_2)
    # # a_2 = Reshape(target_shape=(10, 64, 64, 1))(x)
    #
    # x = CustomLossLayer()(a_1)
    # x = Flatten()(x)
    # x = RepeatVector(n=64)(x)
    # x = Permute((2, 1))(x)
    # x = Reshape(target_shape=(10, 64, 64, 64))(x)
    # mul_1 = multiply([out_4, x])
    # out_5 = UpSampling3D(size=(1, 2, 2))(mul_1)

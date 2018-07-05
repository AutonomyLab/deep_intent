from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Layer
from keras import backend as K
K.set_image_dim_ordering('tf')
from config_classifier import ATTN_COEFF

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


class KLDivergenceLayer(Layer):
    def __init__(self, lamda=1, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(**kwargs)
        self.lamda = lamda

    # def build(self, input_shape):
    #     Create a trainable weight variable for this layer.
        # super(KLDivergenceLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def kld_loss(self, y1, y2, lamda):
        y_true = K.clip(y1, K.epsilon(), 1)
        y_pred = K.clip(y2, K.epsilon(), 1)
        kld_loss = K.sum(y_true * K.log(y_true / y_pred), axis=-1)

        return lamda * kld_loss

    def call(self, inputs):
        y1 = inputs[0]
        y2 = inputs[1]
        loss = self.kld_loss(y1, y2, self.lamda)
        self.add_loss(loss, inputs=inputs)
        # We do use this output.
        return y2

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

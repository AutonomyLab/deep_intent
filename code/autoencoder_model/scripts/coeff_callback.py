from __future__ import absolute_import
from __future__ import print_function

from keras.callbacks import Callback
from keras import backend as K
import numpy as np

if K.backend() == 'tensorflow':
    import tensorflow as tf

class CoeffCallback(Callback):
    """Learning rate scheduler.

    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
    """

    def __init__(self, schedule):
        self.schedule = schedule

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.schedule(epoch)
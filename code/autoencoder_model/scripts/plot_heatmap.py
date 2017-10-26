from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import ast
from config_aa import *

data = np.load(os.path.join(TEST_RESULTS_DIR, 'attention_weights_1.npy'))
data = data[2]
# print (data.shape)
# exit(0)
# Generate some test data
for i in range(10):
    for j in range(data.shape[-1]):
        frame = data[i, :, :, j]
        # frame_1 = data[i, : ,:, 0]
        print (frame.shape)
        frame = np.reshape(frame, data.shape[1:3])

        plt.clf()
        plt.imshow(frame, cmap='hot', interpolation='nearest')
        # plt.imshow(frame, cmap=cm.gray, interpolation='nearest')
        # plt.cbar_axes[1].colorbar()
        plt.savefig(os.path.join(TEST_RESULTS_DIR, 'plot_' + str(i) + '_' + str(j) + '.png'))


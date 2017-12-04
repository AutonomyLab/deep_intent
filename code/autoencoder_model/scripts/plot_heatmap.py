from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import ast
from config_ac import *

def plot_heatmap(attn_layer, epoch, vid_num):
    print (attn_layer)
    for i in attn_layer:
        gen = i
        gen_name = 'gen' + str(gen)
        data = np.load('./../zhora/history/attention_weights_' + gen_name + '_' + str(epoch) + '.npy')
        data = data[vid_num]
        for i in range(10):
            for j in range(data.shape[-1]):
                frame = data[i, :, :, j]
                # frame_1 = data[i, : ,:, 0]
                print (frame.shape)
                frame = np.reshape(frame, data.shape[1:3])

                plt.clf()
                plt.imshow(frame, cmap='binary', interpolation='nearest')
                # plt.imshow(frame, cmap=cm.gray, interpolation='nearest')
                plt.colorbar()
                # plt.cbar_axes[1].colorbar()
                plt.savefig(os.path.join(TEST_RESULTS_DIR, 'plot_' + gen_name + '_' + str(i) + '_' + str(j) + '.png'))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn_layers', nargs='+', type=int)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--vid_num", type=int, default=2)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # print (type(args.attn_layers))
    # print (args.epoch)
    # print (args.vid_num)
    plot_heatmap(attn_layer=args.attn_layers, epoch=args.epoch, vid_num=args.vid_num)

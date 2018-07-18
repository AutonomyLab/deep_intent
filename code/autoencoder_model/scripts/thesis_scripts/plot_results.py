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
import os
from config_r16 import TEST_RESULTS_DIR

def plot_heatmap(attn_layer, epoch, vid_num, file):
    print (attn_layer)
    for i in attn_layer:
        gen = i
        gen_name = 'gen' + str(gen)
        data = np.load(file)
        if file == "None":
            data = np.load('./../zhora/history/attention_weights_' + gen_name + '_' + str(epoch) + '.npy')
        data = data[vid_num]
        for i in range(data.shape[0]):
            for j in range(data.shape[-1]):
                frame = data[i, :, :, j]
                # frame_1 = data[i, : ,:, 0]
                print (frame.shape)
                frame = np.reshape(frame, data.shape[1:3])

                plt.clf()
                plt.imshow(frame, cmap='binary', interpolation='nearest')
                # plt.imshow(frame, cmap=cm.gray, interpolation='nearest')
                # plt.colorbar()
                # plt.cbar_axes[1].colorbar()
                plt.axis('off')
                plt.savefig(os.path.join(TEST_RESULTS_DIR, 'plot_' + gen_name + '_' + str(i) + '_' + str(j) + '.png'),
                            transparent=True)


def plot_err_variation(values, index, dc_value, filename):
    plt.clf()
    plt.axis([1, 17, min(values), max(values)])
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], values, 'b')
    plt.plot([dc_value]*(len(values) + 1), 'r--')
    plt.grid()
    plt.savefig(os.path.join(TEST_RESULTS_DIR + '/graphs/', 'plot_' + filename + '_' + str(index) + '.png'),
                transparent=False)


def errorbars(data):
    means = np.mean(data, axis=0)
    l_error = abs(means-np.min(data, axis=0))
    h_error = abs(means-np.max(data, axis=0))
    plt.clf()
    plt.errorbar()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="None")
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
    plot_heatmap(attn_layer=args.attn_layers, epoch=args.epoch, vid_num=args.vid_num, file=args.file)

# res = np.load('../DRRK16/test_results/graphs/values/1256_mae.npy')
# rescheck = np.load('../NRNK16/test_results/graphs/values/1256_mae.npy')
# dilation = np.load('../NRRK16/test_results/graphs/values/1256_mae.npy')
# kernel = np.load('../NRNN16/test_results/graphs/values/1256_mae.npy')
# rev = np.load('../DRNK16/test_results/graphs/values/1256_mae.npy')
# res_mean = np.mean(res, axis=0)
# rescheck_mean = np.mean(rescheck, axis=0)
# dilation_mean = np.mean(dilation, axis=0)
# kernel_mean = np.mean(dilation, axis=0)
# rev_mean = np.mean(rev, axis=0)

# import math
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
#
# # def myreadlines(f, newline):
# #   buf = ""
# #   while True:
# #     while newline in buf:
# #       pos = buf.index(newline)
# #       yield buf[:pos]
# #       buf = buf[pos + len(newline):]
# #     chunk = f.read(4096)
# #     if not chunk:
# #       yield buf
# #       break
# #     buf += chunk
#
# # plt.rcParams['font.family'] = 'serif'
# plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# plt.rc('text', usetex=True)
#
# # plt.rcParams['font.serif'] = 'Ubuntu'
# # plt.rcParams['font.monospace'] = 'Ubuntu Mono'
# plt.rcParams['font.size'] = 10
# plt.rcParams['axes.labelsize'] = 10
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['xtick.labelsize'] = 8
# plt.rcParams['ytick.labelsize'] = 8
# plt.rcParams['legend.fontsize'] = 10
# plt.rcParams['figure.titlesize'] = 12
# plt.style.use('seaborn-white')
#
# #   if (prefixStr == "Results/200Robots500Boxes/star_switch/star_switch"):
# #     lines = plt.plot(stepNumbers, avgAvgDists)
# #     plt.setp(lines, color='#3E6386', linewidth=1)
# #     plt.fill_between(stepNumbers, [a-b for a,b in zip(avgAvgDists, stdDevs)], [a+b for a,b in zip(avgAvgDists, stdDevs)], facecolor="#9DB3C8", alpha=0.7)
# #   else:
# #     lines = plt.plot(stepNumbers, avgAvgDists)
# #     plt.setp(lines, color='#430029', linewidth=1)
# #     plt.fill_between(stepNumbers, [a-b for a,b in zip(avgAvgDists, stdDevs)], [a+b for a,b in zip(avgAvgDists, stdDevs)], facecolor="#C986AF", alpha=0.7)
# #   ax = plt.gca()
# #   ax.set_yticks(np.arange(0, math.ceil(max(avgAvgDists)), 0.50))
# # plt.xlabel('World Steps')
# # plt.ylabel('Mean Box Distance from Goal Shape (m)')
# # plt.title("Avg Distance from Boxes to \nStar Shaped Goal with and without Switching\n200 Robots 500 Boxes")
# # #plt.fill_between(stepNumbers, [a-b for a,b in zip(avgAvgDistsOut, stdDevsOut)], [a+b for a,b in zip(avgAvgDistsOut, stdDevsOut)], facecolor="#9DB3C8")
# # red_patch = mpatches.Patch(color='#3E6386', label='With Switching')
# # blue_patch = mpatches.Patch(color='#430029', label='Without Switching')
# # plt.legend(handles=[blue_patch, red_patch])
# # ax.set_facecolor([0.975, 0.975, 0.975])
# # plt.savefig('/home/abignell/Pictures/push_pics/starstyle.pdf', bbox_inches='tight')

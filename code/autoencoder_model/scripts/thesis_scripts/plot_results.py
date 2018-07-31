from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import numpy as np
import numpy.random
import math
import argparse
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import cv2
import ast
import json
import os
# from config_rendec16 import TEST_RESULTS_DIR

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


def errorbars(data, model, colour='#3E6386'):

    plt.clf()

    # Setup plot appearances
    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],
                 'monospace':['Computer Modern']})

    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.6
    plt.rcParams['grid.color'] = [0.8, 0.8, 0.8]
    # plt.style.use('seaborn-white')

    meanErr = np.mean(data, axis=0)
    frames = np.linspace(1, 16, 16)
    stdDevs = np.std(data, axis=0)
    # dcVal = [meanErr[-1]] * len(meanErr[0:16])
    # dcDev = [stdDevs[-1]] * len(stdDevs[0:16])
    #
    # dcLine = plt.plot(frames, dcVal)
    # plt.setp(dcLine, color='#D49A6A', linewidth=1.5, linestyle='-.')
    # plt.fill_between(frames, [a - b for a, b in zip(dcVal, dcDev)],
    #                  [a + b for a, b in zip(dcVal, dcDev)], facecolor="#C986AF", alpha=0.7)


    lines = plt.plot(frames, meanErr[0:16], 'o')
    plt.setp(lines, color=colour, linewidth=1, linestyle='-')
    plt.fill_between(frames, [a - b for a, b in zip(meanErr[0:16], stdDevs[0:16])],
                     [a + b for a, b in zip(meanErr[0:16], stdDevs[0:16])], facecolor="#DBF4FF", alpha=0.7)

    ax = plt.gca()
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    # ax.set_yticks(np.arange(0, math.ceil(max(meanErr[0:16])), 0.08))
    # plt.grid()
    ax.yaxis.grid()
    # ax.xaxis.grid()

    plt.xlabel('Predicted frame numbers', fontsize=14)
    plt.ylabel('$l_{1}$ loss', fontsize=14)
    # plt.title("Temporal $l_{1}$ loss Variation")

    ax.set_facecolor([1, 1, 1])
    plt.savefig('/local_home/JAAD_Dataset/thesis/plots/' + model.lower() + '-tem.pdf', bbox_inches='tight')


def errorbars_all(models, y, colours):

    plt.clf()
    # Setup plot appearances
    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],
                 'monospace':['Computer Modern']})

    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 0.5
    # plt.rcParams['axes.color'] = 'white'
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.6
    plt.rcParams['grid.color'] = [0.8, 0.8, 0.8]
    # plt.style.use('seaborn-white')
    model_names = ['Res', 'EnDec', 'Res-EnDec', 'Segment', 'Undilated', 'Unreversed', 'Conv3D']

    for i in range(len(models)):
        data = np.load('/local_home/JAAD_Dataset/thesis/plots/data/mae/1256_mae_' + models[i].lower() + '.npy')

        meanErr = np.mean(data, axis=0)
        frames = np.linspace(1, 16, 16)
        stdDevs = np.std(data, axis=0)

        lines = plt.plot(frames, meanErr[0:16], 'o')
        plt.setp(lines, color=colours[i], linewidth=1, linestyle='-')
        plt.text(16.5, y[i], model_names[i], fontsize=14, color=colours[i])

        # plt.fill_between(frames, [a - b for a, b in zip(meanErr[0:16], stdDevs[0:16])],
        #                  [a + b for a, b in zip(meanErr[0:16], stdDevs[0:16])], facecolor="#DBF4FF", alpha=0.7)

    ax = plt.gca()
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    # ax.set_yticks(np.arange(0, math.ceil(max(meanErr[0:16])), 0.08))
    # plt.grid()
    ax.yaxis.grid()
    # ax.xaxis.grid()

    plt.xlabel('Predicted frame numbers', fontsize=14)
    plt.ylabel('$l_{1}$ loss', fontsize=14)
    # plt.title("Temporal $l_{1}$ loss Variation")

    # red_patch = mpatches.Patch(color='#3E6386', label='With Switching')
    # blue_patch = mpatches.Patch(color='#430029', label='Without Switching')
    # plt.legend(handles=[red_patch])
    # ax.set_facecolor([0.975, 0.975, 0.975])
    # ax.set_facecolor([0.99, 0.99, 0.99])
    ax.set_facecolor([1, 1, 1])
    plt.show()
    plt.savefig('/local_home/JAAD_Dataset/thesis/plots/all-tem.pdf', bbox_inches='tight')


def trainval_plot(data, model, colour='#3E6386'):

    plt.clf()

    # Setup plot appearances
    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],
                 'monospace':['Computer Modern']})

    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.6
    plt.rcParams['grid.color'] = [0.8, 0.8, 0.8]
    # plt.style.use('seaborn-white')

    trainLoss = []
    valLoss = []
    for i in range (len(data)):
        line = json.loads(data[i])
        trainLoss.append(line['train_loss'])
        valLoss.append(line['val_loss'])

    epochs = np.linspace(1, 20, 20)

    valLine = plt.plot(epochs, valLoss[0:20], 'o')
    plt.setp(valLine, color='#D49A6A', linewidth=1.5, linestyle='-')
    # plt.text(20.5, valLoss[19], 'Validation Loss', fontsize=14, color=colour)


    trainLine = plt.plot(epochs, trainLoss[0:20], 'o')
    plt.setp(trainLine, color=colour, linewidth=1, linestyle='-')
    # plt.text(20.5, trainLoss[19], 'Training Loss', fontsize=14, color='#D49A6A')

    train_patch = mpatches.Patch(color=colour, label='Training Loss')
    val_patch = mpatches.Patch(color='#D49A6A', label='Validation Loss')
    plt.legend(handles=[train_patch, val_patch])

    ax = plt.gca()
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    # ax.set_xticks(epochs+1)
    # plt.grid()
    ax.yaxis.grid()
    # ax.xaxis.grid()

    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('$l_{1}$ loss', fontsize=14)
    # plt.title("Training and Validation Losses")

    ax.set_facecolor([1, 1, 1])
    plt.savefig('/local_home/JAAD_Dataset/thesis/plots/' + model.lower() + '-train-val-mse.pdf', bbox_inches='tight')


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
    # plot_heatmap(attn_layer=args.attn_layers, epoch=args.epoch, vid_num=args.vid_num, file=args.file)

    # models = ['Res', 'Rescheck', 'Rendec', 'Kernel', 'Dilation', 'Rev', 'Conv']
    models = ['Res', 'Rescheck', 'Rendec', 'Kernel', 'Dilation', 'Rev']
    # colours = ['#58B33D', '#AB78CC', '#5571B9', '#C43731', '#EC8F1B', '#E394D4']
    colours = ['#58B33D', '#AB78CC', '#5571B9', '#C43731', '#EC8F1B', '#B2C900']


    # 98766B
    # D5D5D5
    # B2C900
    #
    # for i in range (len(models)):
    #     data = np.load('/local_home/JAAD_Dataset/thesis/plots/data/mae/1256_mae_' + models[i].lower() + '.npy')
    #     errorbars(data, models[i])


    y = []
    for model in models:
        data = np.load('/local_home/JAAD_Dataset/thesis/plots/data/mae/1256_mae_' + model.lower() + '.npy')
        meanErr = np.mean(data, axis=0)
        stdDevs = np.std(data, axis=0)
        print (model)
        print (meanErr)
        print ((meanErr[15] - meanErr[2])/13)
        print (((meanErr[15] - meanErr[2])/meanErr[2])*100)
        print (stdDevs)
        print (np.mean(stdDevs))
        print (np.mean(data))

        y_val = meanErr[15]
        if model == 'Rescheck':
            y_val = y_val + 0.0015
        if model == 'Rev':
            y_val = y_val + 0.001
        if model == 'Kernel':
            y_val = y_val - 0.001
        if model == 'Dilation':
            y_val = y_val - 0.0035
        if model == 'Res':
            y_val = y_val - 0.006
        if model == 'Rendec':
            y_val = y_val - 0.0055
        # if model == 'Conv':
        #     y_val = y_val - 0.0055

        y.append(y_val)
    errorbars_all(models, y, colours=colours)

    # for i in range (len(models)):
    #     with open('/local_home/JAAD_Dataset/thesis/plots/data/train-val_loss/losses_gen_' + models[i].lower() + '.json', 'r') as f:
    #         data = f.readlines()
    #     trainval_plot(data, models[i])



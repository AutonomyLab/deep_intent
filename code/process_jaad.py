'''
Code for downloading and processing KITTI data (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
Code borrowed from PredNet (Lotter et al. 2017, https://arxiv.org/abs/1605.08104)
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import hickle as hkl
import numpy as np

from jaad_config import *

desired_im_sz = (128, 128)

# Recordings used for validation and testing.
val_recordings = ['video_0015']
test_recordings = ['video_0027', 'video_0219', 'video_0315', 'video_0123']

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

# Create image datasets.
# Processes images and saves them in train, val, test splits.
def process_data():
    splits = {s: [] for s in ['train', 'test', 'val']}
    splits['val'] = val_recordings
    splits['test'] = test_recordings
    not_train = splits['val'] + splits['test']

    VIDEO_DIR = os.path.join(DATA_DIR, 'seq')
    _, folders, _ = os.walk(VIDEO_DIR).next()
    splits['train'] += [f for f in folders if f not in not_train]

    for split in splits:
        im_list = []
        source_list = []  # corresponds to recording that image came from
        for folder in splits[split]:
            im_dir = os.path.join(VIDEO_DIR, folder)
            _, _, files = os.walk(im_dir).next()
            im_list += [im_dir + "/" + f for f in sorted(files)]
            source_list += [folder] * len(files)

        print('Creating ' + split + ' data: ' + str(len(im_list)) + ' images')
        X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
        for i, im_file in enumerate(im_list):
            try:
                im = cv2.imread(im_file, cv2.IMREAD_COLOR)
                X[i] = process_im(im, im_file, desired_im_sz)
                if split=='train':
                    # cv2.imwrite(os.path.join(RESIZED_IMGS_DIR, im_file[len(im_file) - 14:len(im_file)]), X[i])
                    cv2.imwrite(os.path.join(RESIZED_IMGS_DIR, "train/frame_" + str(i+1) + ".png"), X[i])
                if split=='test':
                    cv2.imwrite(os.path.join(RESIZED_IMGS_DIR, "test/frame_" + str(i+1) + ".png"), X[i])
                if split == 'val':
                    cv2.imwrite(os.path.join(RESIZED_IMGS_DIR, "val/frame_" + str(i+1) + ".png"), X[i])
            except cv2.error as e:
                print("Image file being processed: ", im_file)
                print (e)
            except IOError as e:
                print (e)

        hkl.dump(X, os.path.join(DATA_DIR, 'X_' + split + '_128' + '.hkl'))
        hkl.dump(source_list, os.path.join(DATA_DIR, 'sources_' + split + '_128' + '.hkl'))


# resize image
def process_im(im, im_file, desired_sz):
    im = cv2.resize(im, desired_im_sz, interpolation=cv2.INTER_AREA)
    return im

if __name__ == '__main__':
    process_data()
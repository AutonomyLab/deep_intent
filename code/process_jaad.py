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
import xmltodict as xmlparser

from jaad_config import *

desired_im_sz = (128, 208)

# Recordings used for validation and testing.
# val_recordings = ['video_0027', 'video_0028', 'video_0029', 'video_0030', 'video_0031', 'video_0032']
test_recordings = ['video_0227', 'video_0075', 'video_0127', 'video_0176', 'video_0225', 'video_0274',
                   'video_0323', 'video_0027', 'video_0076', 'video_0128', 'video_0177', 'video_0226',
                   'video_0275', 'video_0324', 'video_0028', 'video_0077', 'video_0129', 'video_0178',
                   'video_0227', 'video_0276', 'video_0325', 'video_0029', 'video_0078', 'video_0130',
                   'video_0179', 'video_0228', 'video_0277', 'video_0326', 'video_0030', 'video_0079',
                   'video_0131', 'video_0180', 'video_0229', 'video_0278', 'video_0327', 'video_0031',
                   'video_0080', 'video_0132', 'video_0181', 'video_0230', 'video_0279', 'video_0328']

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

# Create image datasets.
# Processes images and saves them in train, val, test splits.
def process_data():
    splits = {s: [] for s in ['train', 'test']}
    # splits['val'] = val_recordings
    splits['test'] = test_recordings
    # not_train = splits['val'] + splits['test']
    not_train = splits['test']

    VIDEO_DIR = os.path.join(DATA_DIR, 'seq')
    _, folders, _ = os.walk(VIDEO_DIR).next()
    splits['train'] += [f for f in folders if f not in not_train]

    for split in splits:
        im_list = []
        source_list = []  # corresponds to recording that image came from
        annotation_list = []
        frame_number = 1
        for folder in splits[split]:
            xml_filename = os.path.join(XML_DIR, folder + '.xml')
            with open(xml_filename) as file:
                xml_file = xmlparser.parse(file.read())

            im_dir = os.path.join(VIDEO_DIR, folder)
            _, _, files = os.walk(im_dir).next()
            im_list += [im_dir + "/" + f for f in sorted(files)]

            # Get action labels from xml file
            source_list += [folder] * len(files)
            subjects = xml_file['video']['actions'].keys()
            for i in range(1, len(files)+1):
                actions = ""
                for num, subject in enumerate(subjects):
                    action = ""
                    num_actions = len(xml_file['video']['actions'][subject]['action'])
                    # print (type(xml_file['video']['actions'][subject]['action']))
                    if (type(xml_file['video']['actions'][subject]['action'] )is list):
                        for j in range(num_actions):
                            # print (subject, i, str(xml_file['video']['actions'][subject]['action'][j]['@id']))
                            # print ((round(float(xml_file['video']['actions'][subject]['action'][j]['@start_time'])*30)),
                            #        (round(float(xml_file['video']['actions'][subject]['action'][j]['@end_time']) * 30)))
                            if ((i >= (round(float(xml_file['video']['actions'][subject]['action'][j]['@start_time'])*30)))
                                and (i <= (round(float(xml_file['video']['actions'][subject]['action'][j]['@end_time'])*30)))):
                                    if (len(action)==0):
                                        action = action + str(xml_file['video']['actions'][subject]['action'][j]['@id'])
                                    else:
                                        action = action + ',' + str(xml_file['video']['actions'][subject]['action'][j]['@id'])

                    else:
                        # print(subject, i, str(xml_file['video']['actions'][subject]['action']['@id']))
                        # print((round(float(xml_file['video']['actions'][subject]['action']['@start_time']) * 30)),
                        #       (round(float(xml_file['video']['actions'][subject]['action']['@end_time']) * 30)))
                        if ((i >= (
                        round(float(xml_file['video']['actions'][subject]['action']['@start_time']) * 30)))
                            and (i <= (
                            round(float(xml_file['video']['actions'][subject]['action']['@end_time']) * 30)))):
                            if (len(action)==0):
                                action = action + str(xml_file['video']['actions'][subject]['action']['@id'])
                            else:
                                action = action + "," + str(xml_file['video']['actions'][subject]['action']['@id'])


                    # print (action)
                    if(len(action) ==  0):
                        action = "unknown"
                    actions = actions + str(subject) + ':' + action + (", " if (num<len(subjects)-1) else "")

                info = str(folder) + ', ' + "frame_" + str(frame_number) + ', ' + actions
                frame_number = frame_number + 1
                annotation_list.append(info)
                # print (info)
        #
        print('Creating ' + split + ' data: ' + str(len(im_list)) + ' images')
        X = np.zeros((len(im_list),) + (128, 208, 3), np.uint8)
        for i, im_file in enumerate(im_list):
            try:
                im = cv2.imread(im_file, cv2.IMREAD_COLOR)
                X[i] = process_im(im)
                if split=='train':
                    # cv2.imwrite(os.path.join(RESIZED_IMGS_DIR, im_file[len(im_file) - 14:len(im_file)]), X[i])
                    cv2.imwrite(os.path.join(RESIZED_IMGS_DIR, "train/frame_" + str(i+1) + ".png"), X[i])
                if split=='test':
                    cv2.imwrite(os.path.join(RESIZED_IMGS_DIR, "test/frame_" + str(i+1) + ".png"), X[i])
        #         if split == 'val':
        #             cv2.imwrite(os.path.join(RESIZED_IMGS_DIR, "val/frame_" + str(i+1) + ".png"), X[i])
            except cv2.error as e:
                print("Image file being processed: ", im_file)
                print (e)
            except IOError as e:
                print (e)

        hkl.dump(X, os.path.join(RESIZED_IMGS_DIR, 'X_' + split + '_208' + '.hkl'))
        hkl.dump(source_list, os.path.join(RESIZED_IMGS_DIR, 'sources_' + split + '_208' + '.hkl'))
        hkl.dump(annotation_list, os.path.join(RESIZED_IMGS_DIR, 'annotations_' + split + '_208' + '.hkl'))


# resize image
def process_im(im):
    im = cv2.resize(im, (208, 128), interpolation=cv2.INTER_LANCZOS4)
    return im

if __name__ == '__main__':
    process_data()

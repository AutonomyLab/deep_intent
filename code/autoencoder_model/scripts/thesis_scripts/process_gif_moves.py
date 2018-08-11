import sys
import os
import glob
import shutil
from shutil import copyfile

src = '/local_home/JAAD_Dataset/thesis/results/rendec-gifs/imgs/'
src_gif = '/local_home/JAAD_Dataset/thesis/results/rendec-gifs/'
dst = '/local_home/JAAD_Dataset/thesis/public_posts/pred/rendec/'
vids = ['23', '40', '44', '47', '63', '64', '65', '98', '130', '140', '148', '154', '171',
        '183', '223', '225', '234', '247', '263', '275', '284', '298', '353', '361', '362',
        '367', '376', '377', '444', '509', '526', '527', '532', '537', '616', '621', '629',
        '630', '631', '709', '710', '758']


for num in vids:
    if not os.path.exists(dst + num + '/'):
        os.mkdir(dst + num + '/')

    for file in glob.glob(src + 'vid_' + num + '_frame_*.png'):
    # for file in glob.glob(src + 'vid_' + num + '.gif'):
        print(file)
        shutil.copy(file, dst + '/' + num + '/')
        # copyfile(src, dst)

    gif_src = src_gif + 'vid_' + num + '.gif'
    dst_gif = dst + num + '/' + 'vid_' + num + '.gif'
    print (gif_src)
    print (dst_gif)
    copyfile(gif_src, dst_gif)


import os
import cv2
import argparse
import imageio

def create_gif(filenames, duration, vid_num):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    video_name = "vid_" + str(vid_num) + ".gif"
    imageio.mimsave(os.path.join(GIF_DIR, video_name), images, duration=duration)


def strip(image, img_size, vid_len):
    n_rows = image.shape[0]
    n_cols = image.shape[1]
    n_horizontal_imgs = n_cols/img_size
    n_vertical_imgs = n_rows/img_size
    frame_num = 1
    vid_num = 1
    fps = 30
    duration = 1 / fps
    filenames = []

    for i in range(n_vertical_imgs):
        for j in range(n_horizontal_imgs):
            img = image[i*img_size:(i+1)*img_size, j*img_size:(j+1)*img_size]
            filename = "vid_" + str(vid_num) + "_frame_" + str(frame_num) + ".png"
            cv2.imwrite(os.path.join(GIF_IMG_DIR, filename), img)
            filenames.append(os.path.join(GIF_IMG_DIR, filename))
            if frame_num == vid_len:
                create_gif(filenames=filenames, duration=duration, vid_num=vid_num)
                filenames = []
                vid_num = vid_num + 1
                frame_num = 0
            frame_num = frame_num + 1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="None")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--vid_len", type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    GIF_DIR = '../gifs/'
    if not os.path.exists(GIF_DIR):
        os.mkdir(GIF_DIR)

    GIF_IMG_DIR = '../gifs/imgs/'
    if not os.path.exists(GIF_IMG_DIR):
        os.mkdir(GIF_IMG_DIR)

    args = get_args()
    try:
        im = cv2.imread(args.file, cv2.IMREAD_COLOR)
    except cv2.error as e:
        print("Image file being processed: ", args.file)
        print (e)
    except IOError as e:
        print (e)

    strip(image=im, img_size=args.img_size, vid_len=args.vid_len)
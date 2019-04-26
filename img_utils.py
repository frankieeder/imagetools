import numpy as np
import os
import cv2
import time

def count_frames(path, override=False):
    """Found function for counting frames in video using OpenCV.
    https://www.pyimagesearch.com/2017/01/09/count-the-total-number-of-frames-in-a-video-with-opencv-and-python/
    """
    # grab a pointer to the video file and initialize the total
    # number of frames read
    video = cv2.VideoCapture(path)
    total = 0

    # if the override flag is passed in, revert to the manual
    # method of counting frames
    if override:
        total = count_frames_manual(video)
        # otherwise, let's try the fast way first
    else:
        # lets try to determine the number of frames in a video
        # via video properties; this method can be very buggy
        # and might throw an error based on your OpenCV version
        # or may fail entirely based on your which video codecs
        # you have installed
        try:
            # check if we are using OpenCV 3
            if is_cv3():
                total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            # otherwise, we are using OpenCV 2.4
            else:
                total = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

        # uh-oh, we got an error -- revert to counting manually
        except:
            total = count_frames_manual(video)

    # release the video file pointer
    video.release()

    # return the total number of frames in the video
    return total


def count_frames_manual(video):
    # initialize the total number of frames read
    total = 0

    # loop over the frames of the video
    while True:
        # grab the current frame
        (grabbed, frame) = video.read()

        # check to see if we have reached the end of the
        # video
        if not grabbed:
            break

        # increment the total number of frames read
        total += 1

    # return the total number of frames in the video file
    return total

def get_num_frames(source):
    if SOURCE_TYPE == SOURCE_VIDEO:
        return get_num_frames(source)
    dirpath, dirnames, filenames = next(os.walk(source))
    return len(filenames)


def source_iter(source):
    if SOURCE_TYPE == SOURCE_VIDEO:
        return video_iter(source)
    return picture_iter(source)


def video_iter(video):
    success, image = video.read()
    while success:
        yield image
        success, image = video.read()


def picture_iter(im_folder):
    dirpath, dirnames, filenames = next(os.walk(im_folder))
    for f in filenames:
        im = cv2.imread(dirpath + "/" + f)
        yield im

class VideoIterator():

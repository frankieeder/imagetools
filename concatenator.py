import numpy as np
import os
import cv2
import argparse

SOURCE_TYPE = 0
SOURCE_VIDEO = 1
SOURCE_FOLDER = 0

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


def concatenator(source, region, print_status=False):
    sections = []
    num_frames = get_num_frames(source)
    if print_status:
        print("Starting Concatenation")
    for c, im in enumerate(source_iter(source)):
        if print_status:
            print(f"Processing {c}/{num_frames}")
        sections.append([im[region]])
    if print_status:
        print("Concatenating...")
    final = np.concatenate(sections, axis=1)
    if print_status:
        print("Done\n")
    return final


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


def concatenate(folder_dirs, print_status=False, out_dir="./out"): #TODO: Implement support for multiple folders/files
    for folder in folder_dirs:
        if print_status:
            print("Starting Folder(")

class SliceMaker(object):
  def __getitem__(self, item):
    return item

def main(source,
         out_file,
         video=False, #TODO: Implement video (will need class structure)
         length=-1, #TODO: Implement input length
         random_selection=False, #TODO: Implement random selection of frames
         random_section=False, #TODO: Implement random starting section (if not random selection)
         resize=False, out_w = 1920, out_h = 1080, #TODO: Implement output resizing
         multiple=False #TODO: Process multipe folders/files at once
         ):
    assert length < get_num_frames(source)
    SOURCE_TYPE = SOURCE_VIDEO if video else SOURCE_FOLDER
    final = concatenator(source, SliceMaker()[:,650], print_status=True)
    cv2.imwrite(out_file, final)

main(
    source="./jonah",
    out_file='./jonah.png'
)

#TODO: Implement command line args
"""if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some video or a list of images.')
    parser.add_argument('source', nargs=1)
    parser.add_argument('--random')
    parser.add_argument('--video', dest='source_type', action='store_const',
                        const=SOURCE_VIDEO, default=SOURCE_FOLDER,
                        help='use video file as input (default: use folder filled with images as input)')
    parser.add_argument('--video', dest='source_type', action='store_const',
                        const=SOURCE_VIDEO, default=SOURCE_FOLDER,
                        help='use video file as input (default: use folder filled with images as input)')
    parser.parse_args("X.mp4 --random --video".split())"""


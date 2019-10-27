import numpy as np
import os
import cv2
import time

class ImageStepper:
    def __init__(self, source):
        self.source = source

    def __len__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.source)

class ImageSource(ImageStepper):
    def __init__(self, source_path):
        self.source_path = source_path
        self.source = None  # NOTE: Subclasses must define a source

class SimpleFolderIterator(ImageSource):
    def __init__(self, source_path, img_suffix='.jpg'):
        super().__init__(source_path)
        self.source = self.picture_iter(self.source_path)
        self.img_suffix = img_suffix

    def picture_iter(self, im_folder):
        dirpath, dirnames, filenames = next(os.walk(im_folder))
        filenames = sorted(filenames)
        for f in filenames:
            if f[-len(self.img_suffix):] in self.img_suffix:
                im = cv2.imread(dirpath + "/" + f)
                yield im

    def __len__(self):
        dirpath, dirnames, filenames = next(os.walk(self.source_path))
        return len(filenames)

# TODO: Finish this source
class RandomFolderIterator(ImageSource):
    random_selection = False  # TODO: Implement random selection of frames
    random_section = False  # TODO: Implement random starting section (if not random selection)

class VideoIterator(ImageSource):
    def __init__(self, source_path):
        super().__init__(source_path)
        self.video = cv2.VideoCapture(self.source_path)
        self.source = self.video_iter()

    def __len__(self):
        return self.count_frames()

    def video_iter(self):
        success, image = self.video.read()
        while success:
            yield image
            success, image = self.video.read()

    def count_frames(self, manual_count=False):
        """Found function for counting frames in video using OpenCV.
        https://www.pyimagesearch.com/2017/01/09/count-the-total-number-of-frames-in-a-video-with-opencv-and-python/
        """
        # grab a pointer to the video file and initialize the total
        # number of frames read

        total = 0

        # if the override flag is passed in, revert to the manual
        # method of counting frames
        if manual_count:
            total = self.count_frames_manual()
            # otherwise, let's try the fast way first
        else:
            # lets try to determine the number of frames in a video
            # via video properties; this method can be very buggy
            # and might throw an error based on your OpenCV version
            # or may fail entirely based on your which video codecs
            # you have installed
            try:
                '''
                # check if we are using OpenCV 3
                if is_cv3():
                    total = int(video_file.get(cv2.CAP_PROP_FRAME_COUNT))

                # otherwise, we are using OpenCV 2.4
                else:
                '''
                total = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            # uh-oh, we got an error -- revert to counting manually
            except e:
                total = self.count_frames_manual()

        # release the video file pointer
        self.video.release()

        # return the total number of frames in the video
        return total

    def count_frames_manual(self):
        # initialize the total number of frames read
        total = 0

        # loop over the frames of the video
        while True:
            print(total)
            # grab the current frame
            (grabbed, frame) = self.video.read()

            # check to see if we have reached the end of the
            # video
            if not grabbed:
                break

            # increment the total number of frames read
            total += 1

        # return the total number of frames in the video file
        return total

# ROUGH:
class VideoBlockIterator(VideoIterator):
    def __init__(self, source_path, block_length):
        super().__init__(source_path)
        self.block_length = block_length

    def __len__(self):
        return (self.count_frames() // block_length) + 1

    def video_iter(self):
        all_frames = super().video_iter()
        frame_buffer = []
        for i, frame in enumerate(all_frames):
            frame_buffer.append(frame)
            if i % self.block_length == 0:
                frame_block = np.stack(frame_buffer)
                yield frame_block
                del frame_block
                frame_buffer = []


class SliceMaker(object):
  def __getitem__(self, item):
    return item

def mat2gray(A):
    out = np.zeros(A.shape, np.double)
    normalized = cv2.normalize(A, out, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    return out

def isKthBitOne(n, k):
    """Returns true if the k'th bit of n is 1."""
    return n & (1 << k)

def largestEvenLTE(n):
    """Returns the largest even number less than or equal to the input n."""
    return int(n - (n % 2))

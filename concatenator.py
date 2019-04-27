from img_utils import *
import numpy as np
import os
import cv2
import time
import argparse

class Concatenator(ImageStepper):
    SOURCE_VIDEO = 1
    SOURCE_FOLDER = 0
    def __init__(self,
                 source,
                 length=-1,  # TODO: Implement input length
                 resize=False, out_w=1920, out_h=1080,  # TODO: Implement output resizing
                 region=SliceMaker()[:,:],
                 print_status=False
                 ):
        self.source = source
        self.concat_length = concat_length
        self.resize = resize
        self.out_w = out_w
        self.out_h = out_h
        self.region = region
        self.print_status = print_status
        # TODO: Could implement an iterator for file inputs to concatenator

    def __next__(self):
        if self.print_status:
            start = time.clock()
            print("Starting Concatenation")
        for c in range(self.concat_length):
            im = next(self.source)
            if self.print_status:
                print(f"Processing {c}/{self.concat_length}")
            section = np.expand_dims(im[self.region], axis=1)
            if c == 0:
                final = section
            else:
                final = np.concatenate([final, section], axis=1)
        if self.print_status:
            print(f"Done\n in {time.clock() - start} seconds.")
        return final

    def __iter__(self):
        return self

def main(source,
         out_file,
         video=False, #TODO: Implement video (will need class structure)
         length=1, #TODO: Implement input length
         random_selection=False, #TODO: Implement random selection of frames
         random_section=False, #TODO: Implement random starting section (if not random selection)
         resize=False, out_w=1920, out_h=1080, #TODO: Implement output resizing
         multiple=False #TODO: Process multipe folders/files at once
         ):




main(
    source="./jonah",
    out_file='./jonah.png',
    resize=True,
)
if __name__ == "__main__":
    images = SimpleFolderIterator("./jonah")
    concat = Concatenator(
        source=images
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


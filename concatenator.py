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
                 concat_length=-1,  # TODO: Implement input length
                 resize=False, out_w=1920, out_h=1080,  # TODO: Implement output resizing
                 region=SliceMaker()[:,1],
                 print_status=False,
                 interpolation=cv2.INTER_NEAREST
                 ):
        self.source = source
        self.concat_length = concat_length if concat_length > 0 else len(source)
        self.resize = resize
        self.out_w = out_w
        self.out_h = out_h
        self.region = region
        self.print_status = print_status
        self.interpolation = interpolation
        # TODO: Could implement an iterator for file inputs to concatenator

    def __next__(self):
        if self.print_status:
            start = time.clock()
            print("Starting Concatenation")
        sections = []
        for c in range(self.concat_length):
            im = next(self.source)
            if self.print_status:
                print(f"Processing {c}/{self.concat_length}")
            #section = im[self.region]
            section = np.expand_dims(im[self.region], axis=1)
            sections.append(section)
        final = np.concatenate(sections, axis=1)
        if self.print_status:
            print(f"Done\n in {time.clock() - start} seconds.")
        if self.resize:
            final = cv2.resize(
                src=final,
                dsize=(self.out_w, self.out_h),
                interpolation=self.interpolation
            )
        return final

    def __iter__(self):
        return self


if __name__ == "__main__":
    images = SimpleFolderIterator("/Volumes/My Passport for Mac/Pictures/Design/PROJECTS/55 (Concatenations II)/Video Approach/video jpegs/jonah")
    concat = Concatenator(
        source=images,
        print_status=True,
        region=SliceMaker()[:,960],
        resize=True
    )
    
    for c, im in enumerate(concat):
        cv2.imwrite(f"./jonah_{c}.jpg", im)


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


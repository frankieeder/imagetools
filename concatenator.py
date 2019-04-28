from img_utils import *
import numpy as np
import os
import cv2
import time
from itertools import islice
import argparse

class Concatenator(ImageStepper):
    def __init__(self,
                 source,
                 concat_length=None,
                 resize=False, out_w=1920, out_h=1080,
                 region=SliceMaker()[:,1],
                 print_status=False,
                 interpolation=cv2.INTER_NEAREST
                 ):
        self.source = source
        self.concat_length = concat_length if concat_length else len(source)
        self.resize = resize
        self.out_w = out_w
        self.out_h = out_h
        self.region = region
        self.print_status = print_status
        self.interpolation = interpolation

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


class CubeConcatenator:
    def __init__(self,
                 source,
                 concat_length=None,
                 resize=False, out_w=1920, out_h=1080,
                 print_status=False,
                 interpolation=cv2.INTER_NEAREST
                 ):
        self.source = source
        self.concat_length = concat_length if concat_length else len(source)
        self.resize = resize
        self.out_w = out_w
        self.out_h = out_h
        self.print_status = print_status
        self.interpolation = interpolation
        datas = []
        for c, im in enumerate(islice(self.source, concat_length)):
            datas.append(np.expand_dims(im, axis=0))
            if print_status:
                print(f"Loading image {c} / {self.concat_length}")
        data = np.concatenate(datas, axis=0)
        data = np.swapaxes(data, 0, 2)
        self.data = data
        self.iterator = (im for im in self.data)

    def __next__(self):
        final = next(self.iterator)
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
    images = SimpleFolderIterator(
        "/Volumes/My Passport for Mac/Pictures/Design/PROJECTS/55 (Concatenations II)/Video Approach/video jpegs/jonah")
    concat = CubeConcatenator(
        source=images,
        print_status=True,
        resize=True
    )
    for c in range(concat.concat_length):
        print(f"Saving image {c}")
        cv2.imwrite(f"./jonah_{c}.jpg", next(concat))


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


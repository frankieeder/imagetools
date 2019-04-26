import numpy as np
import os
import cv2
import time
import argparse

class RandomConcatenator:
    SOURCE_VIDEO = 1
    SOURCE_FOLDER = 0
    def __init__(self,
                 source,
                 out_file,
                 video=False,  # TODO: Implement video (will need class structure)
                 length=-1,  # TODO: Implement input length
                 random_selection=False,  # TODO: Implement random selection of frames
                 random_section=False,  # TODO: Implement random starting section (if not random selection)
                 resize=False, out_w=1920, out_h=1080,  # TODO: Implement output resizing
                 multiple=False  # TODO: Process multipe folders/files at once
                 ):
        self.source = source
        # TODO: Could implement an iterator for file inputs to concatenator
        self.source_type = self.SOURCE_VIDEO if video else self.SOURCE_FOLDER

    def __next__(self):
        return concatenate()

    def __iter__(self):
        return self




def concatenator(source, region, print_status=False):
    num_frames = get_num_frames(source)
    if print_status:
        print("Starting Concatenation")
    for c, im in enumerate(source_iter(source)):
        if print_status:
            print(f"Processing {c}/{num_frames}")
        section = np.expand_dims(im[region], axis=1)
        if c == 0:
            final = section
        else:
            final = np.concatenate([final, section], axis=1)
    if print_status:
        print("Done\n")
    return final



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
         resize=False, out_w=1920, out_h=1080, #TODO: Implement output resizing
         multiple=False #TODO: Process multipe folders/files at once
         ):
    assert length < get_num_frames(source)
    SOURCE_TYPE = SOURCE_VIDEO if video else SOURCE_FOLDER
    start = time.clock()
    final = concatenator(source, SliceMaker()[:,960], print_status=True)
    print(time.clock() - start)
    if resize:
        final = cv2.resize(src=final, dsize=(out_w, out_h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_file, final)

main(
    source="./jonah",
    out_file='./jonah.png',
    resize=True,
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


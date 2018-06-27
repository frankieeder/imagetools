from PIL import Image
import numpy as np
import glob
import os
import sys

class ColumnWriter:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        num_images = len([name for name in os.listdir(input_dir)])
        print("Number of images to process:",num_images)
        self.final_img = Image.new("RGB", (6000, num_images))
        self.final_img_px = self.final_img.load()

    def apply_kernel(self, kernel):
        """Applies the input kernel to each image in self.input_dir and
        writes the resulting column to self.final_img.
        """
        #print("Images to process:", images)
        r = 0
        for d in glob.glob(os.path.join(self.input_dir, '*')):
            print("pre-open")
            im = Image.open(d)
            print("pre-apply_kernel")
            kernel(self, im, r)
            print("done with row", r)
            r += 1

    def middle_row(self, img, r):
        """Writes the middle column of img into columnc c of self.final_img.
        """
        w, h = img.size
        imgpx = img.load()
        for x in range(w):
            self.final_img_px[x,r] = imgpx[x, (h/2)]

    def write_to_file(self, file_name = "/final"):
        self.final_img.save(self.input_dir + file_name + ".jpg", "JPEG")

class RowWriter:
    """Probably obsolete, does not account for the fact that images are sideways.
    """
    def __init__(self, input_dir):
        self.input_dir = input_dir
        num_images = len([name for name in os.listdir(input_dir)])
        print(num_images)
        self.final_img = Image.new("RGB", (num_images, 4000))
        self.final_img_px = self.final_img.load()

    def middle_column(self, img, c):
        """Writes the middle column of img into columnc c of self.final_img.
        """
        w, h = img.size
        imgpx = img.load()
        for y in range(h):
            self.final_img_px[c,y] = imgpx[(w/2), y]

    def average_column(self, img, c):
        w, h = img.size
        imgpx = img.load()
        for y in range(h):
            rowsum = (0, 0, 0)
            for x in range(w):
                rowsum = [rowsum[i] + imgpx[x, y][i] for i in range(3)]
            rowavg = tuple(val//w for val in rowsum)
            print(rowavg)
            self.final_img_px[c,y] = rowavg


    def apply_kernel(self, kernel):
        """Applies the input kernel to each image in self.input_dir and
        writes the resulting column to self.final_img.
        """
        c = 0
        for d in glob.glob(os.path.join(self.input_dir, '*')):
            #print("pre-open")
            im = Image.open(d)
            #print("pre-apply_kernel")
            kernel(self, im, c)
            print("done with column", c)
            c += 1

    def collapse_image_horizontally(self, img, resample=Image.BOX):
        """Collapses the input Image img to a column of width 1, using Bilinear
        interpolation by default.
        """
        old_dims = img.size
        new_dims = (1, old_dims[1])
        return img.resize(new_dims, resample)

    def write_column(self, col, i):
        """Writes the input image col into the i'th column of self.final_img.
        """
        assert i in range(self.final_img.size[0] + 1), "Input i should be in the range of the width of the final image"
        self.final_img.paste(col, (i, 0))

    def write_to_file(self, file_name = "/final"):
        self.final_img.save(self.input_dir + file_name + ".jpg", "JPEG")

class ArithmeticResizeCascade:
    def __init__(self, img_dir, resample = Image.BICUBIC):
        self.img_dir = img_dir
        im = Image.open(img_dir)
        im.show()
        w, h = im.size
        #im = im.resize((300, h), resample)
        #w, h = im.size
        img_w = w // 50
        self.final_img = Image.new("RGB", (img_w*(img_w-1)//2, h))
        this_width = w
        paste_point = 0
        while this_width:
            this_img = im.resize((this_width, h), resample)
            self.final_img.paste(this_img, (paste_point, 0))
            paste_point += this_width
            print(this_width)
            this_width -= 50

    def write_to_file(self):
        self.final_img.save(self.img_dir + "_resizes.jpg", "JPEG")

def ColumnWriter_test(input):
    img1 = ColumnWriter(input)
    img1.apply_kernel(ColumnWriter.middle_row)
    img1.write_to_file()


def RowWriter_test(input):
    img1 = RowWriter(input)
    img1.apply_kernel(RowWriter.middle_column)
    img1.write_to_file()

def ArithmeticResizeCascade_test():
    im = ArithmeticResizeCascade("/Volumes/My Passport for Mac/Movies/TEMP FIELD UNSORTED FOOTAGE/3-24-18_PICS_SUNSET & NIGHT/a7/DSC06914.JPG" )
    im.write_to_file()


#ArithmeticResizeCascade_test()
#RowWriter_test()
#ColumnWriter_test()

input = " ".join(sys.argv[1:])
RowWriter_test(input)

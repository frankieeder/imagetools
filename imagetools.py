from PIL import Image
import numpy as np
import glob
import os

class ColumnWriter:

    def __init__(self, input_dir):
        self.input_dir = input_dir
        num_images = len([name for name in os.listdir(input_dir)])
        print(num_images)
        self.final_img = Image.new("RGB", (num_images, 6000))
        c = 0

    def middle_column(self, img):
        """Returns an Image object that is just the middle column of the input
        img.
        """
        w, h = img.size
        return img.crop((w/2, 0, w/2 + 1, h))

    def apply_kernel(self, kernel = ColumnWriter.middle_column):
        """Applies the input kernel to each image in self.input_dir and
        writes the resulting column to self.final_img.
        """
        for d in glob.glob(os.path.join(self.input_dir, '*.JPG')):
            im = Image.open(d)
            col = kernel(self, im)
            self.write_column(col, c)
            print(c)
            c += 1

    def collapse_image_horizontally(self, img, resample=Image.BOX):
        """Collapses the input Image img to a column of width 1, using Bilinear
        interpolation by default.
        """
        #assert isinstance(img, Image)

        old_dims = img.size
        new_dims = (1, old_dims[1])
        return img.resize(new_dims, resample)

    def write_column(self, col, i):
        """Writes the input image col into the i'th column of self.final_img.
        """
        #assert isinstance(col, Image), "Input column should be an Image"
        assert i in range(self.final_img.size[0] + 1), "Input i should be in the range of the width of the final image"
        self.final_img.paste(col, (i, 0))

    def write_to_file(self, file_name = "/final"):
        self.final_img.save(self.input_dir + file_name + ".jpg", "JPEG")

img1 = ColumnWriter(os.getcwd() + "/img_test_1")
img1.apply_kernel()
img1.write_to_file()

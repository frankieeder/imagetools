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

    def middle_column(self, img):
        """Returns an Image object that is just the middle column of the input
        img.
        """
        w, h = img.size
        col = Image.new("RGB", (1, h))
        colpx = col.load()
        imgpx = img.load()
        for y in range(h):
            colpx[0,y] = imgpx[(w/2), y]
        return col

    def apply_kernel(self, kernel):
        """Applies the input kernel to each image in self.input_dir and
        writes the resulting column to self.final_img.
        """
        c = 0
        for d in glob.glob(os.path.join(self.input_dir, '*.JPG')):
            print("pre-open")
            im = Image.open(d)
            print("pre-apply_kernel")
            col = kernel(self, im, col)
            print("pre-write_column")
            self.write_column(col, c)
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

img1 = ColumnWriter( )
img1.apply_kernel(ColumnWriter.middle_column)
img1.write_to_file()

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from skimage import io as skio
from skimage.util import img_as_float32, img_as_ubyte
from skimage.transform import resize


def show_img(im):
    im = skio.imread(im)
    im = img_as_float32(im)
    plt.imshow(im, cmap="gray")
    plt.show()

    print(im.shape)
    print(im.dtype)
    print(im.min(), im.max())

show_img("1/data/cathedral.jpg")




    
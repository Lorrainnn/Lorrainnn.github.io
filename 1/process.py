import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from skimage import io as skio
from skimage.util import img_as_float32, img_as_ubyte
from skimage.transform import resize


def show_img(im):
   
    plt.imshow(im, cmap="gray")
    plt.show()

    print(im.shape)
    print(im.dtype)
    print(im.min(), im.max())



def split_color_channel(im):
    h = im.shape[0] // 3

    B = im[:h, :]
    G = im[h:2*h, :]
    R = im[2*h:3*h, :]

    return B, G, R


im = skio.imread("1/data/cathedral.jpg")
im = img_as_float32(im)
#show_img(im)
print(split_color_channel(im))


    
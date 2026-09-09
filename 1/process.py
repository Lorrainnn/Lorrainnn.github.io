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
    h = im.shape[0]//3

    B = im[:h, :]
    G = im[h:2*h, :]
    R = im[2*h:3*h, :]

    return B, G, R


def l2_score(im1, im2):
    return np.sum((im1 - im2) ** 2)


#boundry crop
def croppping(im, fraction=0.1):
    h, w = im.shape

    crop_y = int(h * fraction)
    crop_x = int(w * fraction)

    return im[
        crop_y:h-crop_y,
        crop_x:w-crop_x
    ]


#The easiest way to align the parts is to exhaustively search over a window of possible displacements (say [-15,15] pixels)
def align_single_l2(mov, ref, search_radius=15):
    best_score = float("inf")
    best_shift = (0, 0)

    reference_crop = croppping(ref)

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):

            shifted = np.roll(
                mov,
                shift=(dy, dx),
                axis=(0, 1)
            )

            shifted_crop = croppping(shifted)

            score = l2_score(
                reference_crop,
                shifted_crop
            )

            if score < best_score:
                best_score = score
                best_shift = (dy, dx)

    return best_shift, best_score

im = skio.imread("1/data/cathedral.jpg")
im = img_as_float32(im)
#show_img(im)
#print(split_color_channel(im))
B, G, R = split_color_channel(im)



naive_rgb = np.dstack([R, G, B])

plt.imshow(naive_rgb)
plt.title("Without Alignment")
plt.show()

g_shift, g_score = align_single_l2(G, B)
r_shift, r_score = align_single_l2(R, B)


G_aligned = np.roll(
    G,
    shift=g_shift,
    axis=(0, 1)
)

R_aligned = np.roll(
    R,
    shift=r_shift,
    axis=(0, 1)
)

result = np.dstack([
    R_aligned,
    G_aligned,
    B
])

plt.imshow(np.clip(result, 0, 1))
plt.title("Aligned Cathedral")
plt.show()
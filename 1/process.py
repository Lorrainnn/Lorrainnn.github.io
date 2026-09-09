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

            #update to best score
            #so basically l2 should be expected to as small as possible

            if score < best_score:
                best_score = score
                best_shift = (dy, dx)

    return best_shift


#this one should be as high as possible (close to 1)
def ncc_score(im1, im2):
    a = im1.ravel()
    b = im2.ravel()

    a = a - np.mean(a)
    b = b - np.mean(b)

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return -np.inf

    return np.dot(a, b) / denominator

def align_single_ncc(moving, reference, search_radius=15):
    best_score = -float("inf")
    best_shift = (0, 0)

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):

            shifted = np.roll(
                moving,
                shift=(dy, dx),
                axis=(0, 1)
            )

            score = ncc_score(
                croppping(shifted),
                croppping(reference)
            )

            if score > best_score:
                best_score = score
                best_shift = (dy, dx)

    return best_shift

def run_single_scale(image_path):
    im = skio.imread(image_path)
    im = img_as_float32(im)

    B, G, R = split_color_channel(im)

    # L2
    g_shift_l2 = align_single_l2(G, B)
    r_shift_l2 = align_single_l2(R, B)

    G_l2 = np.roll(G, g_shift_l2, axis=(0, 1))
    R_l2 = np.roll(R, r_shift_l2, axis=(0, 1))

    result_l2 = np.dstack([
        R_l2,
        G_l2,
        B
    ])



    # NCC
    g_shift_ncc = align_single_ncc(G, B)
    r_shift_ncc = align_single_ncc(R, B)

    G_ncc = np.roll(G, g_shift_ncc, axis=(0, 1))
    R_ncc = np.roll(R, r_shift_ncc, axis=(0, 1))

    result_ncc = np.dstack([
        R_ncc,
        G_ncc,
        B
    ])

    print("\n", image_path)

    print("L2:")
    print("  G (x, y):", (g_shift_l2[1], g_shift_l2[0]))
    print("  R (x, y):", (r_shift_l2[1], r_shift_l2[0]))

    print("NCC:")
    print("  G (x, y):", (g_shift_ncc[1], g_shift_ncc[0]))
    print("  R (x, y):", (r_shift_ncc[1], r_shift_ncc[0]))

    return result_l2, result_ncc


#generate results for jpg with single scale alignment

l2_result, ncc_result  = run_single_scale("1/data/tobolsk.jpg")
plt.figure()
plt.imshow(np.clip(l2_result, 0, 1))
plt.title("tobolsk - L2")
plt.axis("off")
plt.savefig("1/output/l2_tobolsk.jpg")
plt.show()

plt.figure()
plt.imshow(np.clip(ncc_result, 0, 1))
plt.title("tobolsk - NCC")
plt.axis("off")
plt.savefig("1/output/ncc_tobolsk.jpg")
plt.show()

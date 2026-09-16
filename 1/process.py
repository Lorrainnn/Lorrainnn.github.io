import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from skimage import io as skio
from skimage.util import img_as_float32
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

def align_single(moving, reference, style, center=(0, 0), search_radius=15):
    if style == "ncc":
        best_score = -float("inf")
    elif style == "l2":
        best_score = float("inf")
    else:
        raise ValueError("Unknown style")
    
    #init
    best_shift = center
    center_y, center_x = center

    for dy in range(center_y - search_radius, center_y + search_radius + 1):
        for dx in range(center_x - search_radius, center_x + search_radius + 1):

            shifted = np.roll(
                moving,
                shift=(dy, dx),
                axis=(0, 1)
            )
            #we expect ncc close to 1
            if style == "ncc":
                score = ncc_score(
                    croppping(shifted),
                    croppping(reference)
                )
                if score > best_score:
                    best_score = score
                    best_shift = (dy, dx)
            #as small as possible
            elif style == "l2":
                score = l2_score(
                    croppping(shifted),
                    croppping(reference)
                )
                if score < best_score:
                    best_score = score
                    best_shift = (dy, dx)

    return best_shift

#The easiest way to align the parts is to exhaustively search over a window of possible displacements (say [-15,15] pixels)
#def align_single_l2(mov, ref, search_radius=15):


#this one use for p1 jpg single scale output visualization
def run_single_scale(image_path):
    im = skio.imread(image_path)
    im = img_as_float32(im)

    B, G, R = split_color_channel(im)

    # L2
    g_shift_l2 = align_single(G, B, style="l2")
    r_shift_l2 = align_single(R, B, style="l2")

    G_l2 = np.roll(G, g_shift_l2, axis=(0, 1))
    R_l2 = np.roll(R, r_shift_l2, axis=(0, 1))

    result_l2 = np.dstack([
        R_l2,
        G_l2,
        B
    ])


    # NCC
    g_shift_ncc = align_single(G, B, style="ncc")
    r_shift_ncc = align_single(R, B, style="ncc")

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


#p2

def half_size(im):
    return resize(im, (
        im.shape[0] // 2,
        im.shape[1] // 2
    ), anti_aliasing=True)

#500
# In this case, you will need to implement a faster search procedure such as an image pyramid. An image pyramid represents the image at multiple scales (usually scaled by a factor of 2) and the processing is done sequentially starting from the coarsest scale (smallest image) and going down the pyramid, updating your estimate as you go. It is very easy to implement by adding recursive calls to your original single-scale implementation. 
# You should implement the pyramid functionality yourself using appropriate downsampling techniques.
def align_pyramid(mov, ref, style, min_size=200, search_radius=15, refine_radius=3):
    h, w = ref.shape

    #reach the smallest size, adapt single scale alignment
    #end of the recursion call
    if max(h, w)<= min_size:
        return align_single(mov, ref, style=style, center=(0, 0), search_radius=search_radius)

    #downsize
    mov_small = half_size(mov)
    ref_small = half_size(ref)

    #recusive call to align_pyramid
    small_shift = align_pyramid(
        mov_small,
        ref_small,
        style=style,
        min_size=min_size,
        search_radius=search_radius,
        refine_radius=refine_radius
    )

    # upsize the shift
    predicted_shift = (
        2 * small_shift[0],
        2 * small_shift[1]
    )
  
    return align_single(
        mov,
        ref,
        style=style,
        center=predicted_shift,
        search_radius=refine_radius
    )



#generate results for jpg with single scale alignment
#l2_result, ncc_result  = run_single_scale("1/data/tobolsk.jpg")
#plt.figure()
#plt.imshow(np.clip(l2_result, 0, 1))
#plt.title("tobolsk - L2")
#plt.axis("off")
#plt.savefig("1/output/l2_tobolsk.jpg")
#plt.show()

#plt.figure()
#plt.imshow(np.clip(ncc_result, 0, 1))
#plt.title("tobolsk - NCC")
#plt.axis("off")
#plt.savefig("1/output/ncc_tobolsk.jpg")
#plt.show()




#this one use for p2 tiff pyramid scale output visualization
def run_pyramid_scale(image_path):
    im = skio.imread(image_path)
    im = img_as_float32(im)

    B, G, R = split_color_channel(im)

    # L2
    g_shift_l2 = align_pyramid(G, B, style="l2")
    r_shift_l2 = align_pyramid(R, B, style="l2")

    G_l2 = np.roll(G, g_shift_l2, axis=(0, 1))
    R_l2 = np.roll(R, r_shift_l2, axis=(0, 1))

    result_l2 = np.dstack([
        R_l2,
        G_l2,
        B
    ])


    # NCC
    g_shift_ncc = align_pyramid(G, B, style="ncc")
    r_shift_ncc = align_pyramid(R, B, style="ncc")

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


l2_result, ncc_result  = run_pyramid_scale("1/data/melons.tif")
plt.figure()
plt.imshow(np.clip(l2_result, 0, 1))
plt.title("melons - Pyramid L2")
plt.axis("off")
plt.savefig("1/output/l2_pyramid_melons_500.jpg")
plt.show()

plt.figure()
plt.imshow(np.clip(ncc_result, 0, 1))
plt.title("melons - Pyramid NCC")
plt.axis("off")
plt.savefig("1/output/ncc_pyramid_melons_500.jpg")
plt.show()

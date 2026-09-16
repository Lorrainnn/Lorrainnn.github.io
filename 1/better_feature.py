import numpy as np
from skimage import io as skio
from skimage.util import img_as_float32, img_as_ubyte

from process import align_pyramid, split_color_channel

#this part is for cs 280, discussing about the better feature, and how to use it to align the image

#Better features - gradient feature
#here, when under min_size, we will use the gradient feature, 
# expecting emir generate correct img
def gradient_feature(im):
    grad_y, grad_x = np.gradient(im)
    return np.sqrt(grad_x ** 2 + grad_y ** 2)

def run_feature_alignment(image_path, min_size=200):
    im = img_as_float32(skio.imread(image_path))
    B, G, R = split_color_channel(im)

    # gradient features
    B_grad = gradient_feature(B)
    G_grad = gradient_feature(G)
    R_grad = gradient_feature(R)

    g_shift = align_pyramid(
        G_grad, B_grad, style="ncc", min_size=min_size
    )
    r_shift = align_pyramid(
        R_grad, B_grad, style="ncc", min_size=min_size
    )

    # Apply the shifts back to the original color channels.
    G_aligned = np.roll(G, g_shift, axis=(0, 1))
    R_aligned = np.roll(R, r_shift, axis=(0, 1))

    result = np.dstack([R_aligned, G_aligned, B])

    print("G shift:", g_shift)
    print("R shift:", r_shift)

    return result, g_shift, r_shift


# emir_feature, g_shift, r_shift = run_feature_alignment(
#     "1/data/emir.tif",
#     min_size=200
# )
# skio.imsave(
#     "1/output/emir_gradient_ncc.jpg",
#     img_as_ubyte(
#         np.clip(emir_feature, 0, 1)
#     )
# )


#Automatic cropping
#test between border and img
def auto_crop(rgb, max_crop_fraction=0.20):

    # dark = mean_value < 0.15
    # bright = mean_value > 0.6
    # low_variation = std_value < 0.15
    def is_border(mean_val, std_val):
        if mean_val < 0.20 or mean_val > 0.80:
            return std_val < 0.18
        return False
        
    # Convert to grayscale
    gray = np.mean(rgb, axis=2)

    h, w = gray.shape

    row_mean = np.mean(gray, axis=1)
    row_std = np.std(gray, axis=1)

    col_mean = np.mean(gray, axis=0)
    col_std = np.std(gray, axis=0)

    max_y = int(h * max_crop_fraction)
    max_x = int(w * max_crop_fraction)

    
    #top
    top = 0
    while top < max_y and is_border(row_mean[top], row_std[top]):
        top += 1
    #bottom
    bottom = h
    while bottom > h - max_y and is_border(row_mean[bottom - 1],row_std[bottom - 1]):
        bottom -= 1

    #left
    left = 0
    while left < max_x and is_border(col_mean[left],col_std[left]):
        left += 1

    #right
    right = w
    while right > w - max_x and is_border(col_mean[right - 1],col_std[right - 1]):
        right -= 1

    print(
        "crop:",
        "top =", top,
        "bottom =", h - bottom,
        "left =", left,
        "right =", w - right
    )

    return rgb[top:bottom, left:right]



im = skio.imread("1/data/emir.tif")
im = img_as_float32(im)

B, G, R = split_color_channel(im)

g_shift = align_pyramid(G, B, style="ncc", min_size=200)
r_shift = align_pyramid(R, B, style="ncc", min_size=200)

G_aligned = np.roll(G, g_shift, axis=(0, 1))
R_aligned = np.roll(R, r_shift, axis=(0, 1))

aligned_rgb = np.dstack([
    R_aligned,
    G_aligned,
    B
])

# cropped = auto_crop(aligned_rgb)

skio.imsave(
    "1/output/emir_aligned_raw.jpg",
    img_as_ubyte(np.clip(aligned_rgb, 0, 1))
)

# skio.imsave(
#     "1/output/wharf_auto_crop.jpg",
#     img_as_ubyte(np.clip(cropped, 0, 1))
# )

#Automatic contrasting
def auto_contrast(rgb, low_percentile=2, high_percentile=98):
    low = np.percentile(rgb, low_percentile)
    high = np.percentile(rgb, high_percentile)

    if high <= low:
        return rgb.copy()

    result = (rgb - low) / (high - low)

    return np.clip(result, 0, 1)

# rgb = skio.imread("1/output/church_aligned_raw.jpg")
# rgb = img_as_float32(rgb)

# contrast_result = auto_contrast(rgb)

# skio.imsave(
#     "1/output/church_contrast.jpg",
#     img_as_ubyte(contrast_result)
# )

#Automatic white balance
def auto_white_balance(rgb):
    channel_means = np.mean(rgb,axis=(0, 1))

    target_mean = np.mean(channel_means)
    scale = target_mean / channel_means
    result = rgb * scale

    return np.clip(result, 0, 1)

# rgb = skio.imread("1/output/self_portrait_aligned_raw.jpg")
# rgb = img_as_float32(rgb)

# wb_result = auto_white_balance(rgb)

# skio.imsave(
#     "1/output/self_portrait_white_balance.jpg",
#     img_as_ubyte(wb_result)
# )

#Better color mapping
# Better color mapping
def better_color_mapping(rgb):
    #[1.00, -0.05, -0.05],
    #[-0.05, 1.40, -0.05],
    #[-0.05, -0.05, 0.80]
    #channel stength
    color_matrix = np.array([
        [1.10, -0.05, -0.05],
        [-0.05, 1.10, -0.05],
        [-0.05, -0.05, 1.10]
    ])

    h, w, _ = rgb.shape
    pixels = rgb.reshape(-1, 3)
    corrected = pixels @ color_matrix.T
    corrected = corrected.reshape(h, w, 3)

    return np.clip(corrected, 0, 1)

# rgb = skio.imread("1/output/melons_aligned_raw.jpg")
# rgb = img_as_float32(rgb)

# color_result = better_color_mapping(rgb)

# skio.imsave(
#     "1/output/melons_color_mapping.jpg",
#     img_as_ubyte(color_result)
# )

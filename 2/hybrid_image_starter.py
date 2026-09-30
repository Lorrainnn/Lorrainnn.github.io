import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from align_image_code import align_images


def hybrid_image(im1, im2, sigma1, sigma2):
    # high frequency part
    blurred_im1 = gaussian_filter(im1, sigma=(sigma1, sigma1, 0))
    high_freq = im1 - blurred_im1

    # low frequency part
    low_freq = gaussian_filter(im2, sigma=(sigma2, sigma2, 0))

    hybrid = high_freq + low_freq
    return np.clip(hybrid, 0, 1)


def fft_magnitude(image):
    if image.ndim == 3:
        image = np.mean(image, axis=2)

    fft = np.fft.fft2(image)
    fft = np.fft.fftshift(fft)

    return np.log(1 + np.abs(fft))

def to_gray_rgb(image):

    gray = (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    )
    return np.stack([gray, gray, gray], axis=2)


def read_image(path):

    image = plt.imread(path).astype(float)

    if image.max() > 1:
        image = image / 255.0

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    return image


# First load images

# high sf: Nutmeg
im1 = plt.imread(
    "/Users/lorrainnnn/Lorrainnn.github.io/2/data/nutmeg.jpg"
) / 255.0

# low sf: Derek
im2 = plt.imread(
    "/Users/lorrainnnn/Lorrainnn.github.io/2/data/DerekPicture.jpg"
) / 255.0


# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

## You will provide the code below. Sigma1 and sigma2 are arbitrary
## cutoff values for the high and low frequencies


plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(im1)
plt.title("Nutmeg Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(im2)
plt.title("Derek Original")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(im1_aligned)
plt.title("Nutmeg Aligned")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(im2_aligned)
plt.title("Derek Aligned")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "/Users/lorrainnnn/Lorrainnn.github.io/2/data/p2_2_alignment.png",
    dpi=160,
    bbox_inches="tight"
)

plt.show()


# use aligned images from here on

im1 = im1_aligned
im2 = im2_aligned


sigma1 = 4
sigma2 = 12


hybrid = hybrid_image(im1, im2, sigma1, sigma2)


# Nutmeg high-pass
blurred_im1 = gaussian_filter(im1, sigma=(sigma1, sigma1, 0))

high_freq = im1 - blurred_im1


# Derek low-pass
low_freq = gaussian_filter(im2,sigma=(sigma2, sigma2, 0))


high_vis = np.clip(
    high_freq + 0.5,
    0,
    1
)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(high_vis)
plt.title(f"Nutmeg High-pass (sigma = {sigma1})")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(low_freq)
plt.title(f"Derek Low-pass (sigma = {sigma2})")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "/Users/lorrainnnn/Lorrainnn.github.io/2/data/p2_2_components.png",
    dpi=160,
    bbox_inches="tight"
)

plt.show()


plt.imsave(
    "/Users/lorrainnnn/Lorrainnn.github.io/2/data/p2_2_hybrid.png",
    np.clip(hybrid, 0, 1)
)


plt.figure(figsize=(6, 6))
plt.imshow(hybrid)
plt.title("Hybrid Image")
plt.axis("off")
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 8))
plt.subplot(2, 3, 1)
plt.imshow(
    fft_magnitude(im1),
    cmap="gray"
)
plt.title("Nutmeg FFT")
plt.axis("off")


plt.subplot(2, 3, 2)
plt.imshow(fft_magnitude(im2),cmap="gray")
plt.title("Derek FFT")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(fft_magnitude(high_freq),cmap="gray")
plt.title("Nutmeg High-pass FFT")
plt.axis("off")


plt.subplot(2, 3, 4)
plt.imshow(fft_magnitude(low_freq),cmap="gray")

plt.title("Derek Low-pass FFT")
plt.axis("off")


plt.subplot(2, 3, 5)
plt.imshow(fft_magnitude(hybrid),cmap="gray")

plt.title("Hybrid FFT")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "/Users/lorrainnnn/Lorrainnn.github.io/2/data/p2_2_fft.png",
    dpi=160,
    bbox_inches="tight"
)

plt.show()


#bell and whistles

# Color only in low frequencies
high_gray = to_gray_rgb(
    high_freq
)

hybrid_color_low = np.clip(
    high_gray + low_freq,
    0,
    1
)


# Color only in high frequencies
low_gray = to_gray_rgb(low_freq)
hybrid_color_high = np.clip(high_freq + low_gray,0,1)
# Color in both
hybrid_color_both = np.clip(high_freq + low_freq,0,1)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(hybrid_color_low)
plt.title("Color in Low Frequencies")
plt.axis("off")


plt.subplot(1, 3, 2)
plt.imshow(hybrid_color_high)
plt.title("Color in High Frequencies")
plt.axis("off")


plt.subplot(1, 3, 3)
plt.imshow(hybrid_color_both)
plt.title("Color in Both")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    "/Users/lorrainnnn/Lorrainnn.github.io/2/data/p2_2_color.png",
    dpi=160,
    bbox_inches="tight"
)

plt.show()

#C1

custom1_high = read_image(
    "/Users/lorrainnnn/Lorrainnn.github.io/2/data/lor.jpg"
)

custom1_low = read_image(
    "/Users/lorrainnnn/Lorrainnn.github.io/2/data/bf.jpg"
)


c1_high, c1_low = align_images(
    custom1_high,
    custom1_low
)


c1_high_sigma = 4
c1_low_sigma = 10


custom1_hybrid = hybrid_image(c1_high,c1_low,c1_high_sigma,c1_low_sigma)


plt.figure(figsize=(10, 5))


plt.subplot(1, 2, 1)
plt.imshow(custom1_high)
plt.title("High-frequency Input")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(custom1_low)
plt.title("Low-frequency Input")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "/Users/lorrainnnn/Lorrainnn.github.io/2/data/p2_2_custom1_inputs.png",
    dpi=160,
    bbox_inches="tight"
)

plt.show()

plt.imsave(
    "/Users/lorrainnnn/Lorrainnn.github.io/2/data/p2_2_custom1.png",
    np.clip(custom1_hybrid, 0, 1)
)

plt.figure(figsize=(6, 6))
plt.imshow(custom1_hybrid)
plt.title("Custom Hybrid 1")
plt.axis("off")

plt.tight_layout()

plt.show()

#c2

custom2_high = read_image(
    "/Users/lorrainnnn/Lorrainnn.github.io/2/data/nintedo.jpeg"
)
custom2_low = read_image(
    "/Users/lorrainnnn/Lorrainnn.github.io/2/data/nessie.png"
)


c2_high, c2_low = align_images(
    custom2_high,
    custom2_low
)



c2_high_sigma = 4
c2_low_sigma = 10


custom2_hybrid = hybrid_image(c2_high, c2_low, c2_high_sigma, c2_low_sigma)


plt.figure(figsize=(10, 5))


plt.subplot(1, 2, 1)
plt.imshow(custom2_high)
plt.title("High-frequency Input")
plt.axis("off")


plt.subplot(1, 2, 2)
plt.imshow(custom2_low)
plt.title("Low-frequency Input")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    "/Users/lorrainnnn/Lorrainnn.github.io/2/data/p2_2_custom2_inputs.png",
    dpi=160,
    bbox_inches="tight"
)

plt.show()


plt.imsave(
    "/Users/lorrainnnn/Lorrainnn.github.io/2/data/p2_2_custom2.png",
    np.clip(custom2_hybrid, 0, 1)
)


plt.figure(figsize=(6, 6))
plt.imshow(custom2_hybrid)
plt.title("Custom Hybrid 2")
plt.axis("off")
plt.tight_layout()
plt.show()
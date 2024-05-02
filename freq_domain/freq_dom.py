import cv2
import numpy as np
import skimage as ski
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import exposure

# # Read the image
img = cv2.imread('Task5/Fig5.05(a).jpg', 0)

# Compute the Fourier transform
f_transform = np.fft.fft2(img)
f_transform_shifted = np.fft.fftshift(f_transform)
magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
filtered_transform = f_transform_shifted

#mask
rows, cols = img.shape
mask = np.ones((rows, cols), np.uint8)

# Center and radius of the circular exclusion
center_x, center_y = cols // 2, rows // 2
exclude_horizontal = 45#minor and major, could be either way
exclude_vertical = 46
exclude_angle = 0  #no rotation
#apply ellipse to mask
cv2.ellipse(mask, (center_x, center_y), (exclude_horizontal, exclude_vertical), exclude_angle, 0, 360, 0, -1)

# Invert the mask to have 0 in the elliptical region and 1 elsewhere
mask = 1 - mask
# Perform the multiplication with the mask
filtered_transform = filtered_transform * mask
#new magnitude spectrum after mask
magnitude_spectrum_filtered = np.log(np.abs(filtered_transform) + 1)
# Compute the inverse Fourier transform
filtered_img = np.fft.ifft2(np.fft.ifftshift(filtered_transform)).real

#gaussian blur to remove artifacting
img_gaussian = ndimage.gaussian_filter(filtered_img, 3)
#gamma to remove even more artifacting
img_gamma = exposure.adjust_gamma(img_gaussian, 1.7)
#rescale intensities to both remove artifacting and enhance quality
p2, p98 = np.percentile(img_gamma, (17, 99))
img_rescale = exposure.rescale_intensity(img_gamma, in_range=(p2, p98))
#sharpen image to deal with the excess blurring
img_sharp = ski.filters.unsharp_mask(img_rescale, radius=1, amount=14.0)
#adjust the log to bring up the darks just a bit
img_log = exposure.adjust_log(img_sharp, gain=1)
#Final product is of fairly decent quality with minor artifacting
img_processed = img_log

def plot_img(image, ax, title):
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

# Create the figure and subplots
fig, axes = plt.subplots(1, 5, figsize=(15, 4))

# Plot each image
plot_img(img, axes[0], 'Original')
plot_img(magnitude_spectrum, axes[1], 'Original Magnitude Spectrum')
plot_img(filtered_img, axes[2], 'After Filtering')
plot_img(magnitude_spectrum_filtered, axes[3], 'Filtered Magnitude Spectrum')
plot_img(img_processed, axes[4], 'After processing')

# Adjust layout
fig.tight_layout()
# plt.figure()
# plt.imshow(img_processed, cmap='gray')
plt.show()

"""
Note, majority of the code used here and during this was created by ChatGPT

With a simple google search I found that the noise on this image was a sinusoidal periodic type of noise.
There even exists a Stack Overflow discussion on this particular noise.
Unfortunately, it wasn't particularly useful in solving this task, the only useful info was that the image (I'm assuming we're using a similar/some form of copy of the image) could not be perfectly converted back to its original form like in the book.


At first I tried the lowpass filter from Astronaut_on_the_Moon.py.
This worked kind of, but, left the image with lines across the horizontal (This made the image look like it was printed on cardboard).

Afterwards, I studied the magnitude spectrum, which is confusing and most information regarding it seems to be only semi-correct.
It is said that handling bright spots in the spectrum (excluding the middle spot), usually by creating black boxes on top of them to deal with the noise,
this does not appear to be the truth in this case.

(In the following three, the noise became more scrambled than anything else)
- I first noticed this problem when I used a grid system to block out all of the excess bright spots, yet the end result still had significant noise in it.
- Next, changing the previous grid code a bit, a form of notch filtering was achieved with somehow worse results.
- Just in case, I tried notch filtering with manually created small black circles to block out many of the bright spots. This too showed that there wasn't much change to the noise.


I went back to the lowpass filter, and tried gaussian blur and median filtering. At this point in time this did not work.
side note: At some point during my filtering, I noticed that the black background of space had turned gray)


Once I moved to a more custom band-reject filter (an elliptical filter), things started to work better after some adjustments.
Note: The artifacting here is best described as corrugated cardboard (assume that the ridges are where the image is "printed" on and they are thinly and tightly packed together).
- If the values are each set to around 50 or above, then horizontal line artifacting will appear (will also appear if excluded_vertical is more than 1 unit larger than horizontal)
- If the horizontal has a greater value, then vertical line artifacting will appear.
- If the values are below 50 and roughly equal (+-1 unit), then a relatively fine image is acquired with even artifacting 
- If the values dip below 40, then the image gets increasingly blurrier

There is still a lot of artifacting plaquing the image. At this point there is nothing else to do in the frequency domain, so we revert back to a normal image using the inverse fourier.
- A gaussian blur cleans up a significant portion of the artifacting, especially the black background (void of space) (anything above a kernel size of 3 becomes too blurry)
- A gamma and a value rescaling adjustment helps even more with cleaning up the artifacting and creating definition
- A sharpen fixes excess blurring. Anything above 1 for the radius will over sharpen and bring back a lot of the artifacting. The amount has diminishing returns after 14 and can start to oversharpen
- A log adjustment brings up the overall image brightness without affecting the bright spots too much (anything above 1 for the gain is too much). 

Finally, the major steps are plotted, with first the original noise image and its magnitude spectrum, then the filtered image and its magnitude spectrum and lastly the processed image with all of the adjustments made.

In conclusion, in this task, no matter the filter used in the frequency domain, there did not appear to be a way to clean up image from all of the noise and artifacting. Instead, it was required to work in both the frequency and spatial domains.
"""
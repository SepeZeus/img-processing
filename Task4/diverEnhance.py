import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import exposure
import skimage as ski
import matplotlib
import cv2

img = ski.io.imread('Task4/Diver_noise.png', as_gray=False)#mpimg.imread('Task4/Diver_noise.png')
img = ski.img_as_float(img)

matplotlib.rcParams['font.size'] = 8
def plot_img_and_hist(image, axes, bins=256): #less bins=more simplistic, more bins= more detailed, gotta balance
    """Plot an image along with its histogram and cumulative histogram.
    """
    ax_img, ax_hist = axes

    # Display image
    ax_img.imshow(image, cmap='gray')
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    return ax_img, ax_hist


#remove salt and pepper noise
img_medfilt = ndimage.median_filter(img,size=3)#medfilt(img_gaussian, 3)#ski.restoration.denoise_nl_means(img_eqhist, h=1.*sigma_est, fast_mode=False, patch_size=5,patch_distance=3, channel_axis=-1)
#remove gaussian noise
img_gaussian =  cv2.GaussianBlur(img_medfilt, (7,7), 0)

p2, p98 = np.percentile(img, (12, 95))
#rescale intensities
img_rescale = exposure.rescale_intensity(img_gaussian, in_range=(p2, p98)) #map values below 78.0 to 0.0 and values above 129.0 to 1.0

#equalize histogram
img_eqhist = exposure.equalize_adapthist(img_rescale, clip_limit=0.01)#medfilt2d(img, 3)

#brigthen up a bit
img_gamma = exposure.adjust_gamma(img_eqhist, 0.7)
#sharpen
img_sharp = ski.filters.unsharp_mask(img_gamma, radius=1, amount=8.0, channel_axis=-1)
#a bit more brigtening
img_log = exposure.adjust_log(img_sharp, gain=1) #map values below 78.0 to 0.0 and values above 129.0 to 1.0

img_clean = img_log

# Display results
def create_fig(plots):
    fig = plt.figure(figsize=(10, 5))
    axes = np.zeros((2, plots), dtype=object)
    axes[0, 0] = fig.add_subplot(2, plots, 1)
    for i in range(1, plots):
        axes[0, i] = fig.add_subplot(2, plots, 1+i, sharex=axes[0,0], sharey=axes[0,0])
    for i in range(0, plots):
        axes[1, i] = fig.add_subplot(2, plots, (plots+1)+i)
    print(axes)
    
    return axes, fig
axes, fig = create_fig(4)

ax_img, ax_hist, = plot_img_and_hist(img, axes[:, 0])
ax_img.set_title('Original')

ax_img, ax_hist, = plot_img_and_hist(img_medfilt, axes[:, 1])
ax_img.set_title('Medfilt')

ax_img, ax_hist, = plot_img_and_hist(img_gaussian, axes[:, 2])
ax_img.set_title('gaussian')

ax_img, ax_hist, = plot_img_and_hist(img_eqhist, axes[:, 3])
ax_img.set_title('Hist EQ')

axes, fig2 = create_fig(3)

ax_img, ax_hist, = plot_img_and_hist(img_gamma, axes[:, 0])
ax_img.set_title('Gamma')

ax_img, ax_hist, = plot_img_and_hist(img_sharp, axes[:, 1])
ax_img.set_title('Sharpen')

ax_img, ax_hist, = plot_img_and_hist(img_log, axes[:, 2])
ax_img.set_title('Log gain')

fig.tight_layout()
fig2.tight_layout()
plt.figure()
plt.imshow(img_clean, cmap='gray')
plt.show()

#As I do not know enough about noise at this time, I wasn't quite sure what noise was being used
#Using multiple methods I found that the image could be cleaned up quite a lot even without gaussian/medfilt 
#I determined that the image was using both gaussian and salt and pepper noise
#I could not get good results from grayscale image, so I opted for color
#Through experimentation, the closest I could get to seeing the wrist depth meter was around 11-13 meters
#The image seems to be artifacting (if I am not wrong) at the end
#There may be better ways to remove noise and clean up better, but I could not find them at this time (time ran out)
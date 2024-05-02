import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.signal import (medfilt2d, wiener)
from scipy import ndimage
import matplotlib
import math

img = mpimg.imread('Task3/noisyhouse.tif')
#img = img[:, :, 0] # incase of png
img = img/255 #turn values to 0-1

matplotlib.rcParams['font.size'] = 8


def plot_img_and_hist(image, axes, bins=256): #less bins=more simplistic, more bins= more detailed, gotta balance
    """Plot an image along with its histogram and cumulative histogram.
    """
    
    ax_img, ax_hist = axes

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])


    return ax_img, ax_hist

img_medfilt2d = medfilt2d(img, 3)
img_wiener = wiener(img, 5) #first value changes blurryness, second value appears to add some form of black blurry border
img_gaussian = ndimage.gaussian_filter(img, sigma=2) #high value=very blurry

w8 = np.array([[0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]])

# Define a 3x3 kernel/filter for convolution
img_convolve = ndimage.convolve(img, w8, mode='constant') #I tried to copy matlabs imfilter, but it did not seem to work as expected, very gray output

#img_convolve2 = ndimage.convolve(img, kernel, mode='constant')
img_medianfilter = ndimage.median_filter(img, 9) #removes salt and pepper noise nicely when the chunks are big, but struggles finer noise



#custom median filter for removing salt and pepper noise
def custom_medianfilter(img, kernel):
    sizeX = np.shape(img)[0]
    sizeY = np.shape(img)[1]
    img_filtered = np.zeros((sizeX, sizeY))

    edgeX = math.floor(np.shape(kernel)[0]/2)
    edgeY = math.floor(np.shape(kernel)[1]/2)


    for x in range(edgeX, sizeX-edgeX):
        for y in range(edgeY, sizeY-edgeY):
            i = 0
            kernel = np.zeros((7,7))
            #window = np.zeros(kernel[0] * kernel[1], dtype=np.uint8)
            for fx in range(np.shape(kernel)[0]):
                for fy in range(np.shape(kernel)[1]):
                    kernel[fx][fy] = img[x + fx - edgeX][y + fy - edgeY]
                    i+=1
            kernel_values = kernel.flatten()
            kernel_values.sort()
            median_value = kernel_values[len(kernel_values) // 2]
            img_filtered[x][y] = median_value
    return img_filtered


kernel = np.zeros((7,7))
img_custom_medianfilter = custom_medianfilter(img, kernel)

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

print(axes)
ax_img, ax_hist, = plot_img_and_hist(img, axes[:, 0])
ax_img.set_title('Low contrast image')

ax_img, ax_hist, = plot_img_and_hist(img_medfilt2d, axes[:, 1])
ax_img.set_title('Medfilt2d')

ax_img, ax_hist, = plot_img_and_hist(img_wiener, axes[:, 2])
ax_img.set_title('Wiener')

ax_img, ax_hist, = plot_img_and_hist(img_gaussian, axes[:, 3])
ax_img.set_title('gaussian')

axes, fig2 = create_fig(3)

ax_img, ax_hist, = plot_img_and_hist(img-img_convolve, axes[:, 0])
ax_img.set_title('convolve')

ax_img, ax_hist, = plot_img_and_hist(img_medianfilter, axes[:, 1])
ax_img.set_title('medianfilter')


ax_img, ax_hist, = plot_img_and_hist(img_custom_medianfilter, axes[:, 2])
ax_img.set_title('custom medianfilter')

# prevent overlap of y-axis labels
fig.tight_layout()
fig2.tight_layout()


plt.show()

#Depending on image some of them were better than others. The image with the most noticeable differences between filters was the noisyhouse.tif.
#Medfilt2d worked nicely on the left side, but did not do much on the right side.
#Wiener did slightly better on the right side, but failed the left side
#Gaussian got more or less an average of everything (decent lefta and right sides, but not the best)
#Convolve really didn't do anything except add a gray filter on top
#medianfilter and custom median filters both did a decent job of cleaning up the whole iamge (about as good as the gaussian)

#In this image, Medfilt2d with some other filter (probably none of the other filters, except maybe the median filters), should be able to clean up the image quite nicely. 
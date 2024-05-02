import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.signal import convolve2d as conv2
import matplotlib
from skimage import img_as_float
from skimage import exposure
import copy

img = mpimg.imread('Task3/Fig3.20(a).jpg')
img = img/255 #turn values to 0-1

matplotlib.rcParams['font.size'] = 8


def plot_img_and_hist(image, axes, bins=256): #less bins=more simplistic, more bins= more detailed, gotta balance
    """Plot an image along with its histogram and cumulative histogram.
    """
    
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx() #fits the two histograms on the same plot

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
  
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

# Contrast stretching
p2, p98 = np.percentile(img, (12, 80))
#print(p2, p98)
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98)) #map values below 78.0 to 0.0 and values above 129.0 to 1.0

# Equalization
img_eq = exposure.equalize_hist(img)

# Adaptive Equalization
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.27)



# Display results
fig = plt.figure(figsize=(10, 5))
axes = np.zeros((2, 7), dtype=object)
axes[0, 0] = fig.add_subplot(2, 7, 1)
for i in range(1, 7):
    axes[0, i] = fig.add_subplot(2, 7, 1+i, sharex=axes[0,0], sharey=axes[0,0])
for i in range(0, 7):
    axes[1, i] = fig.add_subplot(2, 7, 8+i)

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
ax_img.set_title('Low contrast image')

y_min, y_max = ax_hist.get_ylim()
ax_hist.set_ylabel('Number of pixels')
ax_hist.set_yticks(np.linspace(0, y_max, 5))

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
ax_img.set_title('Contrast stretching')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
ax_img.set_title('Histogram equalization')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
ax_img.set_title('Adaptive equalization')

ax_cdf.set_ylabel('Fraction of total intensity')
ax_cdf.set_yticks(np.linspace(0, 1, 5))

#invet image values
img_neg = 1.0-img #plt.imshow(1-img, cmap="gray")
ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_neg, axes[:, 4])
ax_img.set_title('Negative')

y = 0.27#
c = 2.84 # seems to work similarly to y, except that I could not detect visual change

img_powlaw = c*(img**y)
ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_powlaw, axes[:, 5])
ax_img.set_title('Power Law')

img_log = c*np.log(1+img) #log=natural log
ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_log, axes[:, 6])
ax_img.set_title('Log')

# prevent overlap of y-axis labels
fig.tight_layout()





def filtered_imgs():

    #puts a meshy looking thing onto image, not helpful in this case
    r = np.arange(img.shape[0])
    c = np.arange(img.shape[1])
    # Parameter values for Sinusoidal error
    u0 = 12/(4*np.pi)
    v0 = 8/(4*np.pi)
    A = 0.4
    x, y = np.meshgrid(c, r)
    sin_error = A*np.sin(u0*x + v0*y)
    plt.figure()

    img1 = copy.deepcopy(img) + sin_error

    plt.imshow(img1, clim=(0.0,1.0), cmap="gray")


    sizeX = np.shape(img)[0]
    sizeY = np.shape(img)[1]
    SaltandPepper = np.random.random((sizeX,sizeY)) #noise


    img2 = copy.deepcopy(img)

    d0 = 1200
    filt = np.ones((sizeX,sizeY))
    for i in range(0, sizeX):
        for j in range (0, sizeY):
            if (np.sqrt(((i-sizeX/2)**2 + (j-sizeY/2)**2)) > d0):
                filt[i,j] = 0
    plt.figure()
    
    plt.imshow(img2+SaltandPepper*filt, clim=(0.0,1.0), cmap="gray")

    # Gaussian Lowpass filter, sigma
    img4 = copy.deepcopy(img) # Note the shape of the matrix, x is vertical
    sigma = 200
    filt = np.ones((sizeX,sizeY))
    for i in range(0, sizeX):
        for j in range (0, sizeY):
            filt[i,j] = np.exp(-((i-sizeX/2)**2 + (j-sizeY/2)**2)/(2*sigma**2))

    img4 = img4 + filt*SaltandPepper # Here is now Gaussian or uniform noise, depending np.random() settings, simple!
    plt.figure()
    plt.imshow(img4, clim=(0.0,1.0), cmap="gray")



    # Butterworth Lowpass filter of order n
    n = 4
    img5 = copy.deepcopy(img)
    filt3 = np.ones((sizeX,sizeY))
    for i in range(0, sizeX):
        for j in range (0, sizeY):
            filt3[i,j] = 0.5/(1+(((i-sizeX/16)**2 + (j-sizeY/16)**2)/d0**2)**n)
    plt.imshow(img5+SaltandPepper*filt3, clim=(0.0,1.0), cmap="gray")
    # I could not see differences between the butter and gaussian filters without messing with the values.
    #The mesh is the only one distinctly different from the others

filtered_imgs()
# n=0.2
# i0=15
# j0=80
# d0=20

# fil = np.ones((sizeX,sizeY))
# for i in range(0, sizeX):
#     for j in range (0, sizeY):
#         fil[i,j] = 1/(1+(d0**2/(np.sqrt((i-sizeX/2-i0)**2 + (j-sizeY/2-j0)**2) * np.sqrt((i-sizeX/2+i0)**2 + (j-sizeY/2+j0)**2)))**n)
# fig = plt.figure()
# img2 += fil
# imgplot = plt.imshow(img2, clim=(0.0,1.0), cmap="gray")

plt.show()

#power law(gamma) and histogram equalization alongside adaptive histogram equalization seem to give the most noticeable changes

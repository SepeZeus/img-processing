import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import skimage as ski
import matplotlib
import cv2

#Glassedges.jpg
img = ski.io.imread('Task4/Wrist_xray.jpg', as_gray=True)#mpimg.imread('Task4/Diver_noise.png')
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

#gives quite nicely the edges of all bones
img_sobel = ski.filters.sobel(img)
img_sobelh = ski.filters.sobel_h(img)
img_sobelv = ski.filters.sobel_v(img)

#does better than sobel, but seems to only be due to opencv having a slightly better algorithm
img_gradientx = cv2.Scharr(img, cv2.CV_64F, 1, 0)#ski.filters.scharr(img)#np.gradient(img)
img_gradienty = cv2.Scharr(img, cv2.CV_64F, 0, 1)#ski.filters.scharr(img)#np.gradient(img)
img_gradientx = cv2.convertScaleAbs(img_gradientx)
img_gradienty = cv2.convertScaleAbs(img_gradienty)
img_gradient = cv2.addWeighted(img_gradientx, 0.5, img_gradienty, 0.5, 0)

Laplacian5_filter = np.array([[-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1]])

#With convolve2d, the edges are only faintly visible regardless of whether I use the simple/advanced filter
#and whether the middle number is positive or negative
#The advanced filter with the positive middle number seems to give the "best" results (best is relative)

#With cv2.Laplacian, we can see similar edge outlines as with sobel and scharr

img_laplace = signal.convolve2d(img, Laplacian5_filter, 'same')  #cv2.Laplacian(img, cv2.CV_64F, ksize=7)
#img_laplace = np.uint(np.absolute(img_laplace))

#Compared to the original, I cannot see how these would be particularly useful (these do in a certain sense highlight the bones)
imgSublaplace =  img - img_laplace
laplaceSubImg = img_laplace -img

#Enhances the outlines by a bit
imgScharLap = img_gradient+img_laplace
#enhances laplace's outlines
imgLapSob = img_laplace+img_sobel
#basically inverts the sobel, but a bit grayer 
imgSobSchar = img_sobel-img_gradient

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
ax_img.set_title('Low contrast image')

ax_img, ax_hist, = plot_img_and_hist(img_sobel, axes[:, 1])
ax_img.set_title('sobel')


ax_img, ax_hist, = plot_img_and_hist(img_sobelh, axes[:, 2])
ax_img.set_title('sobel height')


ax_img, ax_hist, = plot_img_and_hist(img_sobelv, axes[:, 3])
ax_img.set_title('sobel vert')

axes, fig2 = create_fig(4)

ax_img, ax_hist, = plot_img_and_hist(img_gradient, axes[:, 0])
ax_img.set_title('Scharr(gradient)')

ax_img, ax_hist, = plot_img_and_hist(img_laplace, axes[:, 1])
ax_img.set_title('laplace')

ax_img, ax_hist, = plot_img_and_hist(imgSublaplace, axes[:, 2])
ax_img.set_title('laplace sub')

ax_img, ax_hist, = plot_img_and_hist(laplaceSubImg, axes[:, 3])
ax_img.set_title('laplace sub2')


axes, fig3 = create_fig(4)

ax_img, ax_hist, = plot_img_and_hist(imgScharLap, axes[:, 0])
ax_img.set_title('Scharr(gradient)+Laplace')

ax_img, ax_hist, = plot_img_and_hist(imgLapSob, axes[:, 1])
ax_img.set_title('laplace+Sobel')

ax_img, ax_hist, = plot_img_and_hist(imgSobSchar, axes[:, 2])
ax_img.set_title('Sobel-Scharr')

# prevent overlap of y-axis labels
fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()

plt.show()

#We can see that all of the edge detection methods can give out the same (very similar) results
#Different libraries can also contribute to how different the results will be
#Sobel was the easiest to implement, and had one of the "best" results out of the box
#Combining these filters can enhance the end product
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage import filters, restoration
from scipy.signal import convolve2d as conv2

img = mpimg.imread('glass.png') # imreads in other modules, too!
img = img[:, :, 0] # But you need NumPy arrays for other calculations

#plt.figure()
#plt.imshow(img, cmap="gray")
# laplace_img = filters.laplace(img) # ksize does not matter, there is a bug where it never gets used, so it always uses a kernel of size (3,3)
# fig8 = plt.figure()
# imgplot = plt.imshow(laplace_img, cmap="gray")
# sharp_img = img + laplace_img # center coeff is positive 5
# print(sharp_img)
# fig = plt.figure()
# imgplot = plt.imshow(sharp_img, clim= (0.1,0.9), cmap="gray")


# Laplacian5_filter = np.array([[1,1,2],[2,-9,2], [1,-1,1]])
# LaplacianSharp_img = conv2(img, Laplacian5_filter, 'same')
# fig = plt.figure()
# imgplot = plt.imshow(LaplacianSharp_img, clim= (0.1,0.9), cmap="gray")

# plt.show()



img = mpimg.imread('Task1/lena.png')
img = img[:,:,0] # some PNG have 3 layers, slice only one for grayscale
fig1 = plt.figure()
imgplot = plt.imshow(img, cmap="gray") # Show original image

img_padded = np.pad(img, 2, 'symmetric') # 2 rows and 2 columns added each side of the original matrix

Sobel_down = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
Sobel_left = Sobel_down.T # Transpose, also .Transpose()
Sobel_up = -1* Sobel_down # Sobels in different orientation
Sobel_right = -1 *Sobel_left

down_img = conv2(img_padded, Sobel_down, 'same') # 2D convolution, also scipy.ndimage.filters.sobel¶
fig2 = plt.figure()
imgplot = plt.imshow(down_img, clim=(0.0,0.9), cmap="gray")
plt.imsave ('Glassedge_down.png', down_img, cmap='gray', vmin = 0.6, vmax = 0.9 )

up_img = conv2(img_padded, Sobel_up, 'same') # 2D convolution, also scipy.ndimage.filters.sobel¶
fig2 = plt.figure()
imgplot = plt.imshow(up_img, clim=(0.0,0.9), cmap="gray")
plt.imsave ('Glassedge_up.png', up_img, cmap='gray', vmin = 0.0, vmax = 0.9 )

left_img = conv2(img_padded, Sobel_left, 'same') # 2D convolution, also scipy.ndimage.filters.sobel¶
fig2 = plt.figure()
imgplot = plt.imshow(left_img, clim=(0.0,0.9), cmap="gray")
plt.imsave ('Glassedge_left.png', left_img, cmap='gray', vmin = 0.0, vmax = 0.9 )

right_img = conv2(img_padded, Sobel_right, 'same') # 2D convolution, also scipy.ndimage.filters.sobel¶
fig2 = plt.figure()
imgplot = plt.imshow(right_img, clim=(0.0,0.9), cmap="gray")
plt.imsave ('Glassedge_right.png', right_img, cmap='gray', vmin = 0.0, vmax = 0.9 )


final_img = np.abs(down_img) - np.abs(left_img)
max_value = final_img.max()
min_value = final_img.min()
#final_img /= max_value # divide by max_value to get 0,…,1.0
final2_img = (final_img - min_value)/(max_value - min_value) # divide by max2_value to get 0,…,1.0
#final_img /= min_value

final2_img = np.abs(up_img) + np.abs(right_img)
max2_value = final2_img.max()
min2_value = final2_img.min()
print(max_value, max2_value, min_value, min2_value)
#final2_img = (final2_img - min2_value)/(max2_value - min2_value) # divide by max2_value to get 0,…,1.0
final2_img /= max2_value

fig6 = plt.figure()
imgplot = plt.imshow(final_img, clim=(0.1,0.4), cmap="gray") # Adjusting the gray levels is by testing, which looks best
plt.imsave ('Glassedge_Sobels.png', final2_img, cmap='gray', vmin = 0.1, vmax = 0.4 )

plt.show()
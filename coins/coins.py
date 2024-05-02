import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import cv2


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
    ax_hist.set_xlim(0, 255)
    ax_hist.set_yticks([])

    return ax_img, ax_hist


img = cv2.imread("Task6/Coins.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #make a grayscale copy
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #change to RGB for displaying purposes
img_detected = img.copy() #detection results are drawn on this

# Apply Gaussian blur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

# Use the HoughCircles method to detect circles in the image
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                            param1=184, param2=32, minRadius=210, maxRadius=240)

def mean_circles(circles, max_offset_x=37, max_offset_y=37, max_overlap_x=112, max_overlap_y=112): #finds mean location and size of circles
    circles_result = []
    circles = np.round(circles[0, :]).astype("int") #float->int
    circles = sorted(circles, key=lambda circle:circle[0])

    x_ref, y_ref = 0,0 #reference value
    cnt = 1
    cur_position = -1 #-1 to deal with out of range after first append
    for (x,y,r) in circles:
        if abs(x-x_ref) < max_offset_x and abs(y-y_ref) < max_offset_y:#if > max offset, then can presume circle is too offset 
            x_res = x + circles_result[cur_position][0]
            y_res = y + circles_result[cur_position][1]
            r_res = r + circles_result[cur_position][2]
            cnt += 1
            circles_result[cur_position] = [x_res,y_res,r_res,cnt]
        elif abs(x-x_ref) < max_overlap_x and abs(y-y_ref) < max_overlap_y: #if > max offset and < max overlap, then can presume too much offset and overlap
            pass
        else:
            cnt = 1
            x_ref = x
            y_ref = y
            circles_result.append([x,y,r, cnt])
            cur_position += 1

    circles_result = [[x//c,y//c,r//c] for (x,y,r,c) in circles_result] #mean values for each circle
    return circles_result

def detect_coin_types(circles): #identifies coins, puts text on img, prints sum and count of each type of coin
    total_value = 0.00
    total_cnts = {"50_Cents_Cnt": 0,
                "1_Euro_Cnt": 0,
                "20_Cents_Cnt": 0
                }
    # Add text to each detected circle
    for i, (x, y, r) in enumerate(circles):
        #radii can have max diff of 2 units before being considered another coin 
        if r >= 221:
            text = '50 Cents'
            total_value += 0.50
            total_cnts["50_Cents_Cnt"] += 1
        elif r >= 218 and r < 221:
            text = '1 Euro'
            total_value += 1
            total_cnts["1_Euro_Cnt"] += 1
        elif r >= 215 and r < 218:
            text = '20 Cents'
            total_value += 0.20
            total_cnts["20_Cents_Cnt"] += 1
        cv2.putText(img_detected, text, (x-280, y-350), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 40, 0), 12, cv2.LINE_AA)

    print(f"Total amount of money: {total_value:.2f} €")
    print(f"Total count of 50 Cent coins: {total_cnts['50_Cents_Cnt']}")
    print(f"Total count of 1 Euro coins: {total_cnts['1_Euro_Cnt']}")
    print(f"Total count of 20 Cent coins: {total_cnts['20_Cents_Cnt']}")
    
if circles is not None:
    circles_mean = mean_circles(circles)
    #Iterate through detected circles and draw them on the image
    for (x, y, r) in circles_mean:
        cv2.circle(img_detected, (x, y), r, (0,255,0), 25)
    detect_coin_types(circles_mean)

# Display results
def create_fig(plots):
    fig = plt.figure(figsize=(10, 5))
    axes = np.zeros((2, plots), dtype=object)
    axes[0, 0] = fig.add_subplot(2, plots, 1)
    for i in range(1, plots):
        axes[0, i] = fig.add_subplot(2, plots, 1+i, sharex=axes[0,0], sharey=axes[0,0])
    for i in range(0, plots):
        axes[1, i] = fig.add_subplot(2, plots, (plots+1)+i)    
    return axes, fig
axes, fig = create_fig(2)

ax_img, ax_hist, = plot_img_and_hist(img, axes[:, 0])
ax_img.set_title('Original')

ax_img, ax_hist, = plot_img_and_hist(img_detected, axes[:, 1])
ax_img.set_title('Detected Coins')

fig.tight_layout()
plt.show()

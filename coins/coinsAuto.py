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
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    return ax_img, ax_hist


img = cv2.imread("Task6/Coins.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #make a grayscale copy
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #change to RGB for displaying purposes
imc = img.copy()

from random import randint
# Apply Gaussian blur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

for i in range(1, 45):
    p1 = randint(50,230)
    p2= randint(15,40)
    d= randint(1, 45)
    # Use the HoughCircles method to detect circles in the image
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=d,
                                param1=p1, param2=p2, minRadius=210, maxRadius=240)

    def mean_circles(circles, max_offset_x, max_offset_y): #finds mean location and size of circles
        circles_result = []
        global b1,b2,b3,b4
        global bp,bpp
        circles = np.round(circles[0, :]).astype("int")
        circles = sorted(circles, key=lambda x:x[0])

        x_ref, y_ref = 0,0 #reference value
        cnt = 1
        cur_position = -1 #first time, values are added to circles_result
        for (x,y,r) in circles:
            if abs(x-x_ref) < max_offset_x and abs(y-y_ref) < max_offset_y:#if >57, then can presume is
                x_res = x + circles_result[cur_position][0]
                y_res = y + circles_result[cur_position][1]
                r_res = r + circles_result[cur_position][2]
                cnt += 1
                circles_result[cur_position] = [x_res,y_res,r_res,cnt]
            elif abs(x-x_ref) < 112 and abs(y-y_ref) < 112:
                pass
            else:
                cnt = 1
                x_ref = x
                y_ref = y
                circles_result.append([x,y,r, cnt])
                cur_position += 1
            
        circles_result = [[x//c,y//c,r//c] for (x,y,r,c) in circles_result]
        if(len(circles_result) == 4):
            print(circles_result[1])
            if (abs(circles_result[0][2] - circles_result[1][2]) < 4 
            and ((circles_result[2][2] < circles_result[0][2])
            and (circles_result[2][2] < circles_result[1][2]))
            and (circles_result[2][2]  > circles_result[3][2])):
                b1 = circles_result[0][2]    
                b2 = circles_result[1][2]
                b3 = circles_result[2][2]
                b4 = circles_result[3][2]
                bp = p1
                bpp = p2

            #print(circles_result)
        return circles_result

    def detect_coins(circles):
        total_value = 0.00
        total_cnts = {"50_Cents_Cnt": 0,
                    "1_Euro_Cnt": 0,
                    "20_Cents_Cnt": 0
                    }
        # Add text to each detected circle
        for i, (x, y, r) in enumerate(circles):
            if r >= 218:
                text = '50 Cents'
                total_value += 0.50
                total_cnts["50_Cents_Cnt"] += 1
            elif r >= 216 and r < 218:
                text = '1 Euro'
                total_value += 1
                total_cnts["1_Euro_Cnt"] += 1
            elif r >= 214 and r < 216:
                text = '20 Cents'
                total_value += 0.20
                total_cnts["20_Cents_Cnt"] += 1
            cv2.putText(img, text, (x-280, y-350), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 40, 0), 12, cv2.LINE_AA)

        # print(f"Total amount of money: {total_value:.2f} €")
        # print(f"Total count of 50 Cent coins: {total_cnts['50_Cents_Cnt']}")
        # print(f"Total count of 1 Euro coins: {total_cnts['1_Euro_Cnt']}")
        # print(f"Total count of 20 Cent coins: {total_cnts['20_Cents_Cnt']}")
        
    print("Attempt:", i, p1, p2, d)
    mask = np.zeros_like(img_gray)
    if circles is not None:
        circles_mean = mean_circles(circles)
        # Iterate through detected circles and draw them on the mask
        # for (x, y, r) in circles_mean:
        #     cv2.circle(mask, (x, y), r, (255), -1)

        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img,contours, -1, (0,255,0),25)
        # detect_coins(circles_mean)
print(b1,b2,b3,b4, bp, bpp, d)
detect_coins(blah)
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
axes, fig = create_fig(2)

ax_img, ax_hist, = plot_img_and_hist(imc, axes[:, 0])
ax_img.set_title('Original')

ax_img, ax_hist, = plot_img_and_hist(mask, axes[:, 1])
ax_img.set_title('Thresholded')

ax_img, ax_hist, = plot_img_and_hist(img, axes[:, 2])
ax_img.set_title('Detected Coins')

fig.tight_layout()
plt.show()

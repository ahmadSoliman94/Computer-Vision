import os 
import numpy as np
import cv2


# ================================= Mean Shift =================================


# read video
cap = cv2.VideoCapture('./video.mp4')

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
x , y , width , height = 300 , 200 , 100 , 50 # simply hardcoded the values
track_window = (x , y , width , height)

# set up the ROI for tracking
roi = frame[y:y+height , x:x+width]
hsv_roi = cv2.cvtColor(roi , cv2.COLOR_BGR2HSV) # convert to HSV color-space
mask = cv2.inRange(hsv_roi , np.array((0. , 60. , 32.)) , np.array((180. , 255. , 255.))) # histogram back projection # 0. , 60. , 32. are the lower bound of the histogram back projection and 180. , 255. , 255. are the upper bound of the histogram back projection

# calculate the histogram
roi_hist = cv2.calcHist([hsv_roi] , [0] , mask , [180] , [0 , 180]) # calculate the histogram of the ROI and use it to calculate the back projection

# normalize the histogram
cv2.normalize(roi_hist , roi_hist , 0 , 255 , cv2.NORM_MINMAX)



# setup the termination criteria , either 10 iteration or move by atleast 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT , 10 , 1)

while True:

    ret , frame = cap.read()

    '''
    hsv = hue saturation value  represents a color model that describes colors (hue or tint) , 
    saturation (purity or amount of hue) and brightness value. 
    Hue ranges from 0-360 , saturation from 0-1 and value from 0-1
    Hue is the best to represent color , so it is used most often in computer vision applications.
    '''

    hsv = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV) # convert to HSV color-space

    # calculate the back projection based on the roi histogram
    dst = cv2.calcBackProject([hsv] , [0] , roi_hist , [0 , 180] , 1) # calculate the back projection based on the roi histogram [hsv] is the source image , [0] is the channel used , roi_hist is the histogram back projection , [0 , 180] is the range of the histogram , 1 is the scale factor

    # apply meanshift to get the new location
    ret , track_window = cv2.meanShift(dst , track_window , term_crit) 

    # draw it on image
    x , y , w , h = track_window
    final_image = cv2.rectangle(frame , (x , y) , (x+w , y+h) , 255 , 3) 

    cv2.imshow('dst' , dst)
    cv2.imshow('final_image' , final_image)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

cap.release()



'''
What is Back Projection? 
It is a way of recording how well the pixels of a given image fit the distribution of pixels in a histogram model.
To make it simpler, for Back Projection, you calculate the histogram model of a feature and then use it to find this feature in an image.

The basic idea behind back projection  is to compare a model or template image with a target image in order to find regions in the target image that resemble the model. 
The model image typically contains the desired object or feature of interest, while the target image is the image in which we want to locate the object.

for more information about Back Projection : https://docs.opencv.org/3.4/da/d7f/tutorial_back_projection.html
'''
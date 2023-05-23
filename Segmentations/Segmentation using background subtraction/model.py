import cv2
import os 


# ============================== Segment images using background subtraction ==============================

'''
Background subtraction is a method to separate foreground from background based on the change in pixel values.
MAth formula:
    I(x, y, t) = I(x, y, t) - I(x, y, t-1) 
    where I(x, y, t) is the pixel value at location (x, y) and t is the current frame.
    t-1 is the previous frame.
'''

# Read video
cap = cv2.VideoCapture('./highway.mp4')

# read the first frame
_ , first_frame = cap.read() 

# Convert to grayscale
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# apply Gaussian blur to the image to remove noise and for better accuracy
prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0) # (5, 5) is the kernel size and 0 is the standard deviation: is the measure of how spread out numbers are.


while True:
    # read the next frame
    ret , frame = cap.read() # ret is a boolean variable that returns true if the frame is available.
    if not ret: 
        break

    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # apply Gaussian blur to the image to remove noise and for better accuracy
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # calculate the difference between the current frame and the previous frame
    difference = cv2.absdiff(prev_gray, gray)

    # calculate the threshold
    _, difference = cv2.threshold(difference, 50 , 255, cv2.THRESH_BINARY)  # 25 is the threshold value, 255 is the max value, cv2.THRESH_BINARY is the threshold type, _ is the threshold value returned by the function

    # show the difference in the image
    cv2.imshow("First frame", first_frame)
    cv2.imshow("Frame", frame)
    cv2.imshow("difference", difference)

    key = cv2.waitKey(30) # wait for 30ms before moving to the next frame
    if key == 27: # press ESC to exit
        break

# release the video capture object
cap.release()
cv2.destroyAllWindows()



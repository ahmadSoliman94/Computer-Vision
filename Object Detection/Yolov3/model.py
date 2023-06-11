import cv2
import numpy as np
import os 


# ======================================== YOLOv3 ========================================

# Load YOLOv3 weights and config file

'''
To download the weights and config file:
1. Go to https://pjreddie.com/darknet/yolo/
2. Download the weights file: yolov3.weights
3. Download the config file: yolov3.cfg
4. Download the class labels file: coco.names

or  use the following commands:
1. wget https://pjreddie.com/media/files/yolov3.weights
2. wget https://pjreddie.com/media/files/yolov3.cfg
3. wget https://pjreddie.com/media/files/coco.names
'''

net = cv2.dnn.readNet("./yolov3.weights", "./yolov3.cfg")

# Load COCO class labels
classes = []

with open("./coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    

# Get layer names
layer_names = net.getLayerNames()


print(net.getUnconnectedOutLayers())
for i in net.getUnconnectedOutLayers():
    print(layer_names[i-1])
    
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

# set the colors for the bounding boxes
colors = np.random.uniform(0, 255, size=(len(classes), 3)) # 3 channels , random colors


# load the image
img = cv2.imread("./animals.jpg")

# resize the image
img = cv2.resize(img, None, fx=0.4, fy=0.4) # 0.4 is the scale factor for both axes s

# get the height and width of the image
height, width, channels = img.shape

# Detecting objects

'''
blob from image: is a function that takes an image as input and does the following:
1. Resizes the image to the specified size
2. Subtracts the mean values from each channel of the image
3. Swaps the channels of the image
4. Creates a blob and returns it

params:
1. image: the input image
2. scalefactor: used to scale the image values.
3. size: the size of the image that the model expects
4. mean: the mean value for each channel
5. swapRB: a boolean value that indicates whether to swap the first and last channels
6. crop: a boolean value that indicates whether to crop the image or not.
'''

blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False) # 416x416 is the size of the image that the model expects 

# set the input for the network
net.setInput(blob)

# run the forward pass to get the output of the output layers
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []

# loop over each of the output layers
for out in outs:
    # loop over each of the detections
    for detection in out:
        # get the class id (label) and confidence (probability) of the current object detection
        scores = detection[5:] # the first 5 elements are the coordinates of the bounding box and the objectness score. The rest are the class probabilities.
        class_id = np.argmax(scores)

        # get the confidence (probability) of the current object detection
        confidence = scores[class_id]

        # check if the confidence is greater than the threshold
        if confidence > 0.2:
            # get the coordinates of the bounding box for the object
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)

            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # get the coordinates of the top left corner of the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h]) # append the coordinates of the bounding box to the boxes list
            confidences.append(float(confidence)) # append the confidence (probability) of the current object detection to confidences list
            class_ids.append(class_id) # append the class id (label) of the current object detection to class_ids list


# apply non-max suppression to eliminate redundant overlapping boxes with lower confidences
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4) # 0.5 is the confidence threshold, 0.4 is the threshold for the IOU (Intersection Over Union)


'''draw the bounding boxes and label on the image'''

font = cv2.FONT_HERSHEY_PLAIN # font for the text

# loop over the bounding boxes
for i in range(len(boxes)):
    if i in indexes: # if the index i exists in the indexes list
        x, y, w, h = boxes[i] # get the coordinates of the bounding box

        label = str(classes[class_ids[i]]) # get the class label
        color = colors[i] # get the color for the bounding box

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2) # draw the bounding box
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3) # put the label text above the bounding box


# show the image
cv2.imshow("Image", img)

# save the image
cv2.imwrite("./output.jpg", img)

# wait until any key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
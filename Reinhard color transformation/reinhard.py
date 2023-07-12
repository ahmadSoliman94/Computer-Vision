''' 
This approach is suitable for stain normalization of pathology images where
the 'look and feel' of all images can be normalized to a template image. 
This can be a good preprocessing step for machine learning and deep learning 
of pathology images. 

'''


import os
import cv2
import numpy as np


# ================================ Reinhard Transform ================================


INPUT_DIR = './input_images/'
OUTPUT_DIR = './output_images/'
TEMPLATE_DIR = './template_images/'

# get the mean and standard deviation

def get_mean_std(img):

    x_mean, x_std = cv2.meanStdDev(img) # get mean and standard deviation
    x_mean = np.hstack(np.around(x_mean, decimals=2)) # hstack: stack arrays in sequence horizontally
    x_std = np.hstack(np.around(x_std, decimals=2)) 
    return x_mean, x_std


# read the input images 

input_images = []

for file in os.listdir(INPUT_DIR):
    if file.endswith('.jpg'):
        input_images.append(file)

for img in input_images:
    input_image = cv2.imread(os.path.join(INPUT_DIR, img))

    # convert the image from RGB to LAB color space
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)

    # get the mean and standard deviation of the  input image
    input_mean, input_std = get_mean_std(input_image)


# read the template image
template_image = cv2.imread(os.path.join(TEMPLATE_DIR, 'template.jpg'))

# convert the template image from RGB to LAB color space
template_lab = cv2.cvtColor(template_image, cv2.COLOR_BGR2LAB)

# get the mean and standard deviation of the template image
template_mean, template_std = get_mean_std(template_lab)


# get the height and width and number of channels of the input image
h, w, c = input_image.shape


# iterate over each channel and perform the color normalization
for i in range(0,h):
    for j in range(0,w):
        for k in range(0,c):
            x = input_image[i,j,k] # get the pixel value of the input image
            x = ((x - input_mean[k]) * (template_std[k] / input_std[k])) + template_mean[k]	 # perform the color normalization
            x = round(x) # round the pixel values
            # check the boundary conditions
            x = 0 if x < 0 else x # if x < 0, then x = 0 else x = x
            x = 255 if x > 255 else x # if x > 255, then x = 255 else x = x
            input_image[i,j,k] = x # assign the new pixel value to the input image

# convert the input images from LAB to RGB color space
input_image = cv2.cvtColor(input_image, cv2.COLOR_LAB2BGR)

# save the output images
cv2.imwrite(os.path.join(OUTPUT_DIR, 'output.jpg'), input_image)

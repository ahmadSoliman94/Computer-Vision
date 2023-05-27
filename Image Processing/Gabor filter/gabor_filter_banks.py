import numpy as np
import cv2
import pandas as pd

# Import required libraries

# read image
img = cv2.imread('./synthetic.jpg')  # Read the image file

# convert to gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale

'''
IF you have a multi-channel image, then extract the channel you want to work on insted of converting to gray scale.
for example, if you have a 3-channel image, then you can extract the first channel as follows:
img_gray = img[:,:,0]
'''

# reshape the image
img2 = img_gray.reshape(-1)  # Reshape the grayscale image into a 1D array

# create a dataframe
df = pd.DataFrame()  # Create an empty DataFrame

# add pixel values to the data frame
df['original_image'] = img2  # Add the grayscale image as a column in the DataFrame with the label 'original_image'

# Generate Gabor features

num = 1  # To count numbers up in order to give Gabor features a label in the data frame
kernels = []  # Create an empty list to hold all kernels that we will generate in a loop

for theta in range(2):  # Define number of thetas
    theta = theta / 4. * np.pi  # Convert theta to the corresponding angle in radians
    
    for sigma in (1, 3):  # Sigma values of 1 and 3
        for lamda in np.arange(0, np.pi, np.pi / 4):  # Range of wavelengths from 0 to pi with step size pi/4
            for gamma in (0.05, 0.5):  # Gamma values of 0.05 and 0.5 . if gamma is close to 1, the Gaussian kernel is almost circular. if gamma is close to 0, the Gaussian kernel is almost elliptical shape. 

                gabor_label = 'Gabor' + str(num)  # Label Gabor columns as Gabor1, Gabor2, etc.

                # create gabor kernel
                ksize = 5  # Size of the Gabor filter (n, n)
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi=0, ktype=cv2.CV_32F)
                kernels.append(kernel)  # Append the generated Gabor kernel to the list

                # Now filter the image and add values to a new column
                fimg = cv2.filter2D(img_gray, cv2.CV_8UC3, kernel)  # Apply the Gabor filter to the grayscale image
                filtered_img = fimg.reshape(-1)  # Reshape the filtered image into a 1D array
                df[gabor_label] = filtered_img  # Add the filtered image values as a new column in the DataFrame with the Gabor label
                print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  # Increment the counter for the Gabor column label

# show images
cv2.imshow('original', img)  # Display the original image
cv2.imshow('filtered', fimg)  # Display the filtered image
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close all the windows

# show the dataframe
print(df.head())  # Display the first few rows of the DataFrame

# save the dataframe as csv file
df.to_csv('./Gabor_features.csv')  # Save the DataFrame as a CSV file

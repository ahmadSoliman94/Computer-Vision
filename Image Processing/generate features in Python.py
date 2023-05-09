import cv2
import pandas as pd

from skimage.filters.rank import entropy
from skimage.filters import sobel
from skimage.morphology import disk

from scipy import ndimage as ndi

############################ Generate Features #################################

# # 1. Load the image
img = cv2.imread('./images/Yeast_Cells.png')

# reshape the image to one dimension
img2 = img.reshape(-1)


# # 2. Convert the image to grayscale
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # 3.  apply the entropy filter
# entropy_img = entropy(img, disk(1)) # disk(1) is a 1 pixel radius

# # 4. Display the entropy image
# cv2.imshow('Original image', img)
# cv2.imshow('Entropy image', entropy_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


##################################################################################

# apply gaussian blur to the image


gaussian = ndi.gaussian_filter(img, sigma=3) # sigma is the radius of the kernel

# reshape the image to one dimension
gaussian2 = gaussian.reshape(-1)

# cv2.imshow('Original image', img)
# cv2.imshow('Gaussian image', gaussian)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



##################################################################################
# apply sobel filter to the image 
sobel = sobel(img) 

# reshape the image to one dimension
sobel2 = sobel.reshape(-1)



# cv2.imshow('Original image', img)
# cv2.imshow('Sobel image', sobel)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


##################################################################################


# create a dataframe to store the features
df = pd.DataFrame()
df['Orginal pixel values'] = img2
print(df.head())

# add the entropy image pixel values as a column to the dataframe
df['Gaussian pixel values'] = gaussian2

# add the sobel image pixel values as a column to the dataframe
df['Sobel pixel values'] = sobel2


df.to_csv('features.csv', index=False)

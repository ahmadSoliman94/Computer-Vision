#Scratch Assay on time series images
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import threshold_otsu

time = 0 
time_list=[]
area_list=[]
path = "./scratch_assay/*.*"
for file in glob.glob(path): # Loop through all images in folder.
    dict={}
    img=io.imread(file)
    entropy_img = entropy(img, disk(3)) # entropy() used to find the entropy of an image.
    thresh = threshold_otsu(entropy_img) # threshold_otsu() used to find the optimal threshold value for an image.
    binary = entropy_img <= thresh # Create binary image
    scratch_area = np.sum(binary == 1) # Calculate area of scratch
    print("time=", time, "hr  ", "Scratch area=", scratch_area, "pix\N{SUPERSCRIPT TWO}") # Print time and area of scratch.
    time_list.append(time)
    area_list.append(scratch_area)
    time += 1

#print(time_list, area_list)
plt.plot(time_list, area_list, 'bo')  # 'bo' is for blue dots.
plt.show()

# to finde slope, intercept

'''
slope represents the rate of change of the dependent variable with respect to the independent variable.
intercept represents the value of the dependent variable when the independent variable is zero.
r_value represents the correlation coefficient.
p_value represents the probability of observing the data if the null hypothesis (that there is no significant linear relationship between the variables) is true.
std_err represents the standard error of the estimate.
'''
slope, intercept, r_value, p_value, std_err = linregress(time_list, area_list) 
print("y = ",slope, "x", " + ", intercept  )
print("R\N{SUPERSCRIPT TWO} = ", r_value**2) # SUPERSCRIPT TWO is for square.
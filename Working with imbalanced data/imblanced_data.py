import numpy as np
import cv2
import pandas as pd
 
img = cv2.imread('./Sandstone_Versa0180_image.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

#Save original image pixels into a data frame. This is our Feature #1.
img2 = img.reshape(-1)
df = pd.DataFrame()
df['Original Image'] = img2

#Generate Gabor features
num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
kernels = []
for theta in range(2):   #Define number of thetas
    theta = theta / 4. * np.pi
    for sigma in (1, 3):  #Sigma with 1 and 3
        for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
            for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
            
                
                gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
#                print(gabor_label)
                ksize=9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                kernels.append(kernel)
                #Now filter the image and add values to a new column 
                fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  #Increment for gabor column label


#Now, add a column in the data frame for the Labels
#For this, we need to import the labeled image
labeled_img = cv2.imread('./Sandstone_Versa0180_mask.png')
#Remember that you can load an image with partial labels 
#But, drop the rows with unlabeled data

labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
labeled_img1 = labeled_img.reshape(-1)
df['Labels'] = labeled_img1

print(df.head())


print(df.Labels.unique())  #Look at the labels in our dataframe
print(df['Labels'].value_counts())


#Define the dependent variable that needs to be predicted (labels)
Y = df["Labels"].values

#Define the independent variables
X = df.drop(labels = ["Labels"], axis=1) 


#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


####################################################################    

# # 1.  Appropriate Model Selection for Imbalanced Data: Random Forest 

# #RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
# model_RF = RandomForestClassifier(n_estimators = 10, random_state = 42)

# # Train the model on training data
# model_RF.fit(X_train, y_train)

# #Test prediction on testing data. 
# prediction_test_RF = model_RF.predict(X_test)

# #ACCURACY METRICS
# print("********* METRICS FOR IMBALANCED DATA *********")
# #Let us check the accuracy on test data
from sklearn import metrics
# print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test_RF))

# (unique, counts) = np.unique(prediction_test_RF, return_counts=True)
# print(unique, counts)

# #Confusion Matrix
from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, prediction_test_RF)
# print(cm)

# #Print individual accuracy values for each class, based on the confusion matrix
# print("Pixel 33 accuracy = ", cm[0,0] / (cm[0,0]+cm[1,0]+cm[2,0]+cm[3,0]))
# print("Pixel 65 accuracy = ",   cm[1,1] / (cm[0,1]+cm[1,1]+cm[2,1]+cm[3,1]))
# print("Pixel 201 accuracy = ",   cm[2,2] / (cm[0,2]+cm[1,2]+cm[2,2]+cm[3,2]))
# print("Pixel 231 accuracy = ",   cm[3,3] / (cm[0,3]+cm[1,3]+cm[2,3]+cm[3,3]))

####################################################################

# # 2. Up-sample minority class
# from sklearn.utils import resample
# print(df['Labels'].value_counts())

# #Separate majority and minority classes
# df_important = df[df['Labels'] == 201]
# df_majority = df.loc[df['Labels'].isin([33, 231])]
# df_minority = df[df['Labels'] == 65]

# # Upsample minority class and other classes separately
# # If not, random samples from combined classes will be duplicated and we run into
# #same issue as before, undersampled remians undersampled.
# df_minority_upsampled = resample(df_minority, 
#                                  replace=True,     # sample with replacement
#                                  n_samples=400000,    # to match average class
#                                  random_state=42) # reproducible results
 
# df_important_upsampled = resample(df_important, 
#                                  replace=True,     # sample with replacement
#                                  n_samples=400000,    # to match average class
#                                  random_state=42) # reproducible results


# # Combine majority class with upsampled minority class
# df_upsampled = pd.concat([df_majority, df_important_upsampled, df_minority_upsampled])
# print(df_upsampled['Labels'].value_counts())

# Y_upsampled = df_upsampled["Labels"].values

# #Define the independent variables
# X_upsampled = df_upsampled.drop(labels = ["Labels"], axis=1) 


# #Split data into train and test to verify accuracy after fitting the model. 
# from sklearn.model_selection import train_test_split
# X_train_upsampled, X_test_upsampled, y_train_upsampled, y_test_upsampled = train_test_split(X_upsampled, 
#                                                                                             Y_upsampled, 
#                                                                                             test_size=0.2, 
#                                                                                             random_state=20)

# #Train again with new upsamples data
# model_RF_upsampled = RandomForestClassifier(n_estimators = 10, random_state = 42)

# # Train the model on training data
# model_RF_upsampled.fit(X_train_upsampled, y_train_upsampled)
# prediction_test_RF_upsampled = model_RF_upsampled.predict(X_test_upsampled)

# print("********* METRICS FOR BALANCED DATA USING UPSAMPLING *********")

# print ("Accuracy = ", metrics.accuracy_score(y_test_upsampled, prediction_test_RF_upsampled))

# cm_upsampled = confusion_matrix(y_test_upsampled, prediction_test_RF_upsampled)
# print(cm_upsampled)

# print("Pixel 33 accuracy = ", cm_upsampled[0,0] / (cm_upsampled[0,0]+cm_upsampled[1,0]+cm_upsampled[2,0]+cm_upsampled[3,0]))
# print("Pixel 65 accuracy = ",  cm_upsampled[1,1] / (cm_upsampled[0,1]+cm_upsampled[1,1]+cm_upsampled[2,1]+cm_upsampled[3,1]))
# print("Pixel 201 accuracy = ",  cm_upsampled[2,2] / (cm_upsampled[0,2]+cm_upsampled[1,2]+cm_upsampled[2,2]+cm_upsampled[3,2]))
# print("Pixel 231 accuracy = ",  cm_upsampled[3,3] / (cm_upsampled[0,3]+cm_upsampled[1,3]+cm_upsampled[2,3]+cm_upsampled[3,3]))

###########################################################

# 3. Generate synthetic data (SMOTE and ADASYN)

# SMOTE: Synthetic Minority Oversampling Technique
#ADASYN: Adaptive Synthetic
# https://imbalanced-learn.org/stable/over_sampling.html?highlight=smote

from imblearn.over_sampling import SMOTE, ADASYN

X_smote, Y_smote = SMOTE().fit_resample(X, Y)  #Beware, this takes some time based on the dataset size
# X_adasyn, Y_adasyn = ADASYN().fit_resample(X, Y)

X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, 
                                                                            Y_smote, 
                                                                            test_size=0.2, 
                                                                            random_state=42)


(unique, counts) = np.unique(Y, return_counts=True)
print("Original data: ", unique, counts)
(unique2, counts2) = np.unique(Y_smote, return_counts=True)
print("After SMOTE: ", unique2, counts2)
#(unique3, counts3) = np.unique(Y_adasyn, return_counts=True)
#print("After ADASYN: ", unique3, counts3)

model_SMOTE = RandomForestClassifier(n_estimators = 10, random_state = 42)
model_SMOTE.fit(X_train_smote, y_train_smote)

prediction_test_smote = model_SMOTE.predict(X_test_smote)

print ("Accuracy = ", metrics.accuracy_score(y_test_smote, prediction_test_smote))

prob_y_test_smote = model_RF.predict_proba(X_test_smote)
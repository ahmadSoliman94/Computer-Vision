import os 
import cv2 
import glob
import imgaug.augmenters as iaa



# ==================== Image Augmentation ====================

# list all images in the folder
#images_path = os.listdir('./images') # 
images = []
for img in glob.glob('./images/*.jpg'):
    img = cv2.imread(img)
    images.append(img)
   

print(images)


# define augmentation
'''
1. Flip: flip the image horizontally or vertically.
2. Affine: rotate, scale, translate, shear the image. tralate is the shift of the image, scale is the zoom in or zoom out of the image, shear is the distortion of the image.
3. multiply: change the brightness of the image.
4. LinearContrast: change the contrast of the image. contrast is the difference in luminance or color that makes an object (or its representation in an image or display) distinguishable.
'''
aug = iaa.Sequential([

    # 1. Flip
    #iaa.Fliplr(0.5), # 50% chance that image is flipped horizontally
    iaa.Flipud(1), # 50% chance that image is flipped vertically

    # 2. Affine
    iaa.Affine(
        #translate_precent={'x':(-0.1, 0.1), 'y':(-0.1, 0.1)}, # translate by -10% to +10% on x- and y-axis
        rotate=(-30, 30), # rotate by -30 to +30 degrees)
        #scale=(0.5, 1.5), # scale image to 50% to 150%)
         ), 

    # 3. Multiply
    #iaa.Multiply((0.5, 1.5)), # change brightness of the image to 50% to 150%



    # 4. LinearContrast
    # iaa.LinearContrast((0.5, 1.4)), # change contrast of the image to 50% to 140% . Enhance the pixel values of the image. for example, if the pixel value is 0.5, then the new pixel value will be 0.5 * 1.4 = 0.7

    # 5. GaussianBlur
    iaa.GaussianBlur(sigma=(0, 3.0)), # blur the image with a sigma of 0 to 3.0
    # Perform a gaussian blur with methods sometims:
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 3.0))), # blur the image with a sigma of 0 to 3.0, blur 50% of the images.


])

# apply augmentation to all images
augmented_images = aug(images=images)

# for i,img  in enumerate(augmented_images):
#     cv2.imshow('image_' + str(i), img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# save augmented images
while True:
    for i, img in enumerate(augmented_images):
        cv2.imwrite('./images/image_' + str(i) + '.jpg', img)
    break
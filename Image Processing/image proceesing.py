#Reading images
import glob
from PIL import Image 


images_path = r"./Image Processing/images/"
img = Image.open(images_path + "test_image.jpg")

# show Images on external default viewer. This can be paint or photo viewer on Windows
img.show() 

# prints format of image 
print(img.format) 
  
# prints mode of image RGB or CMYK
print(img.mode) 

# prints the size of image (wodth, height)
print(img. size)  

# 1. Resize images
small_img = img.resize((200, 300))
small_img.save(images_path + "test_image_small.jpg")  #squished image

# resize() method resizes images to exact value whether it makes sense or not.
#aspect ratio is not maintained so images are squished.
#if you want to keep the aspect ration then use thumbnai() method

img.thumbnail((200, 200))
img.save(images_path + "test_image_small_new.jpg")

print(img.size)

img.thumbnail((1200, 1200))  #doesn't blow up the image, only reduces the size if original is larger.
img.save(images_path + "test_image_small_new1.jpg")  

large_img = img.resize((1200, 1300))
large_img.save(images_path + "test_image_large.jpg")  #enlarged image. 
print(large_img.size)

############################################################################################

# 2. Cropping images
cropped_img = img.crop((0, 0, 300, 300))  #crops from (0,0) to (300,300)
cropped_img.save(images_path + "cropped_img.jpg")


# We can paste image on another image
#this involves copying original image and pasting a second image on it
img1 = Image.open(images_path  + "test_image.jpg")
print(img1.size)
img2 = Image.open(images_path + "monkey.jpg")
print(img2.size)
img2.thumbnail((200, 200))  #Resize in case the image is very large. 

img1_copy = img1.copy()   #Create a copy of the large image
img1_copy.paste(img2, (50, 50))  #Paset the smaller imager image at specified location
img1_copy.save(images_path + "pasted_image.jpg")

############################################################################################

# 3. Rotating images
from PIL import Image 
img = Image.open(images_path + "test_image.jpg")

img_90_rot = img.rotate(90)
img_90_rot.save(images_path + "rotated90.jpg")  #keeps original aspect ratio and dimensions

img_45_rot = img.rotate(45)
img_45_rot.save(images_path + "rotated45.jpg")  #keeps original aspect ratio and dimensions

img_45_rot = img.rotate(45, expand=True)  #Dimensions are expanded to keep  the entire image
img_45_rot.save(images_path + "rotated45.jpg")  


############################################################################################

# 4. Flipping or transposing images


img = Image.open(images_path + "monkey.jpg")  #easy to see that the image is flipped

img_flipLR = img.transpose(Image.FLIP_LEFT_RIGHT)
img_flipLR.save(images_path + "flippedLR.jpg")

img_flipTB = img.transpose(Image.FLIP_TOP_BOTTOM)
img_flipTB.save(images_path + "flippedTB.jpg")

# Color transforms, convert images between L (greyscale), RGB and CMYK

img = Image.open(images_path + "test_image.jpg")

grey_img = img.convert('L')  #L is for grey scale
grey_img.save(images_path + "grey_img.jpg")

################################################################################################

# Here is a way to automate image processing for multiple images.

path =  images_path + "aeroplane/*.*"
for file in glob.glob(path):
    print(file)     # just stop here to see all file names printed
    a = Image.open(file)  # now, we can read each file since we have the full path
    
    rotated45 = a.rotate(45, expand=True)
    rotated45.save(file+"_rotated45.png", "PNG")  



# Image processing libraries:

## 1. Pillow.

## 2. Spicpy: 
### Scipy is a python library that is part of numpy stack. 
### It contains modules for linear algebra, FFT, signal processing and image processing. 
### Not designed for image processing but has a few tools.

## 3. sickit-image

## basic image transformation: 
- ### resize and rescale.
 
<br />

# Edge detection: 
- ## Edge detection filters are image processing techniques used to highlight the boundaries or edges between different regions or objects in an image. The basic idea behind edge detection filters is to identify the areas in the image where there are rapid changes in brightness or color, which typically correspond to object boundaries or edges.

### - There are various types of edge detection filters:
1. ### Sobel filter: This is a gradient-based filter that calculates the first-order derivatives of the image in both the horizontal and vertical directions, and then combines them to produce a gradient magnitude image. The resulting image highlights the edges in the image as bright or dark lines.

2. ### Canny filter: This is a multi-stage edge detection algorithm that applies a Gaussian filter to smooth the image, calculates the gradient magnitude and direction using a Sobel filter, and then applies non-maximum suppression and hysteresis thresholding to refine the edge map and remove noise.

3. ### Laplacian filter: This is a second-order derivative filter that calculates the Laplacian of the image to detect regions of rapid intensity changes. The resulting image highlights both edges and flat regions with high frequency content.

4. ### Prewitt filter: This is another gradient-based filter that calculates the gradient magnitude and direction using a pair of 3x3 convolution kernels. The resulting image highlights the edges in the image as bright or dark lines.


<br />

### __Image deconvolution:__ is a process of restoring the original image from its degraded or blurred version. Image degradation occurs due to various factors such as motion blur, out-of-focus blur, and noise. In image deconvolution, the point spread function (PSF) is used to model the blurring process.

### - The deconvolution process involves the following steps:

1. ### PSF estimation: The first step in image deconvolution is to estimate the PSF of the system that has caused the image degradation. This can be done by measuring the response of the system to a known input or by assuming a certain type of blurring.

2. ### Deconvolution: Once the PSF is estimated, the deconvolution process can be carried out. The deconvolution process involves dividing the degraded image by the estimated PSF in the Fourier domain and then applying an inverse Fourier transform to obtain the deblurred image. However, this process can amplify the noise in the image and lead to artifacts.


<br />

### - Entropy filter: 
- ### can be used to separate regions and can detect subtle variations in the local gray level distribution.

---------------------------
<br />

###  - Filters work by convolution with a moving window called a kernel.
### - Convolution is nothing but multiplication of two arrays of different sizes.
### - The image will be of one size and the kernel with be of a different size, usually much smaller than image.
### - The input pixel is at the centre of the kernel. 
### -  The convolution is performed by sliding the kernel over the image, usually from top left of image.

<br />

- ## ___Linear filters:___  are typically used for smoothing or blurring an image, or for removing noise. Examples of linear filters include the Gaussian filter, the mean filter. These filters work by convolving the image with a kernel or mask that contains weights that are applied to each pixel in the image.

- ## ___Non-linear filters:___ are used for more complex image processing tasks such as edge detection, image segmentation, and feature extraction. Unlike linear filters, non-linear filters use non-linear operations such as thresholding, morphological operations, and nonlinear transformations to manipulate the image. Examples of non-linear filters include the Sobel filter, the Canny edge detector, and the morphological filter and  the median filter. 

<br />

### - __A Gaussian kernel__: is a type of linear filter used in image processing to blur or smooth an image. 

### - Is defiend by the following equation: 
```scss
G(x,y) = (1/2πσ^2) * e^(-(x^2+y^2)/2σ^2)
```

###  NLM: is a non-local means denoising algorithm. It is based on the assumption that similar patches are likely to be found in close proximity in an image.

<br />

# Image Enhancments:
- ### Some times images ack contrast, they appear to be washed out but they still contain information.
- ### We can mathematically process these images and make them look good, more importantly, get them ready for segmentation.
- ###  __Using:__
    1. ### Histogram equalization: is an image processing technique that adjusts the contrast of an image by using its histogram. To enhance the image’s contrast, it spreads out the most frequent pixel intensity values or stretches out the intensity range of the image. By accomplishing this, histogram equalization allows the image’s areas with lower contrast to gain a higher contrast.

<br />

2. ### Adaptive Histogram Equalization (AHE): dividing the image into smaller regions, calculating the histogram of each region, and then applying HE to each region individually. This results in a more localized contrast enhancement, which can better preserve the details of the image.

> ### __NOTE:__ adaptive histogram equalization is better than the ordinary histogram equalization if you want to improve the local contrast and enhance the edges in specific regions of the image.

<br />

3. ### Contrast Limited Adaptive Histogram Equalization (CLAHE): divide an image into small, overlapping tiles, and then apply the HE algorithm to each tile separately. This approach allows for better control of the contrast enhancement in different regions of the image, as the local histogram of each tile can be adjusted independently.

> ###  one of the main problems with traditional HE is that it can amplify noise in regions with low contrast, which can result in unnatural-looking artifacts. To address this problem, CLAHE sets a limit on the amount of contrast enhancement that can be applied to each tile, based on the overall contrast of the image. This limit helps to prevent the amplification of noise and other artifacts, while still allowing for significant contrast enhancement in regions with low contrast.


<br />

# Thresholding:
- ### is a widely used image preprocessing technique that is used to separate regions of an image based on their intensity values. It involves setting a threshold value, which divides the pixel values of the image into two groups: those above the threshold and those below it.

### -  There are several different types of thresholding algorithms:
1. ### Global thresholding: In this technique, a single threshold value is applied to the entire image, regardless of the local variations in intensity.
2. ### Adaptive thresholding: This technique involves dividing the image into small regions and applying a threshold value to each region independently, based on the local intensity values.
3. ### Otsu's thresholding: This technique involves finding the threshold value that maximizes the separation between the foreground and background regions of the image, using a statistical measure known as the "between-class variance."



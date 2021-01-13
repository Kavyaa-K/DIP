# DIP
Q1. Develop a program to display grayscale image using read and write the operation.

**Description**
Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.
**Importance of grayscaling**
Dimension reduction: For e.g. In RGB images there are three color channels and has three dimensions while grayscaled images are single dimensional.
Reduces model complexity: Consider training neural article on RGB images of 10x10x3 pixel.The input layer will have 300 input nodes. On the other hand, the same neural network will need only 100 input node for grayscaled images.
For other algorithms to work: There are many algorithms that are customized to work only on grayscaled images e.g. Canny edge detection function pre-implemented in OpenCV library works on Grayscaled images only.

imread() : is used for reading an image. 
imwrite(): is used to write an image in memory to disk. 
imshow() :to display an image. 
waitKey(): The function waits for specified milliseconds for any keyboard event.
destroyAllWindows():function to close all the windows. cv2. cvtColor() method is used to convert an image from one color space to another For color conversion, we use the function cv2. cvtColor(input_image, flag) where flag determines the type of conversion. For BGR Gray conversion we use the flags cv2.COLOR_BGR2GRAY
np.concatenate: Concatenation refers to joining. This function is used to join two or more arrays of the same shape along a specified axis.

**Program**
import cv2 
import numpy as np 
image = cv2.imread('p4.jpg') 
cv2.imshow('Old', image) 
cv2.waitKey() 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
cv2.imshow('Gray', gray) 
cv2.imwrite('sample.jpg',gray) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

**Output:**
![image](https://user-images.githubusercontent.com/72294293/104423697-6c9a9680-55a4-11eb-81c0-e202500b807c.png)





Q2) Develop the program to perform linear transformation on image. Description

**Program Rotation of the image:**
A)Scaling
**Description**
Image resizing refers to the scaling of images. Scaling comes handy in many image processing as well as machine learning applications. It helps in reducing the number of pixels from an image 
cv2.resize() method refers to the scaling of images. Scaling comes handy in many image processing as well as machine learning applications. It helps in reducing the number of pixels from an image 
imshow() function in pyplot module of matplotlib library is used to display data as an image

**Program:**
import cv2 import numpy as np 
img = cv2.imread('p17.jpg') 
(rows, cols) = img.shape[:2] 
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 120, 1) 
res = cv2.warpAffine(img, M, (cols, rows)) 
cv2.imshow('result', res) 
cv2.imshow('image',img) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

**Output:**
![image](https://user-images.githubusercontent.com/72294293/104424552-88eb0300-55a5-11eb-95f0-6aa41f691bef.png)

B) Rotating of image. 
**Description**
Image rotation is a common image processing routine used to rotate images at any desired angle. This helps in image reversal, flipping, and obtaining an intended view of the image. Image rotation has applications in matching, alignment, and other image-based algorithms. 
OpenCV is a well-known library used for image processing.
cv2.getRotationMatrix2D Perform the counter clockwise rotation
warpAffine() function is the size of the output image, which should be in the form of (width, height). Remember width = number of columns, and height = number of rows.

**Program**
import cv2 
import numpy as np 
img = cv2.imread('p17.jpg') 
(height, width) = img.shape[:2] 
res = cv2.resize(img, (int(width / 2), int(height / 2)), interpolation = cv2.INTER_CUBIC) 
cv2.imshow('result', res) 
cv2.imshow('image',img) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

**Output:**
![image](https://user-images.githubusercontent.com/72294293/104425426-ac627d80-55a6-11eb-8d93-26c109923644.png)


Q3) Develop a program to find the sum and mean of set of image create n number of images and read from directory and perform the operation
**Description**
You can add two images with the OpenCV function, cv. add(), or simply by the numpy operation res = img1 + img2.
The function mean calculates the mean value M of array elements, independently for each channel, and return it:" This mean it should return you a scalar for each layer of you image
The append() method in python adds a single item to the existing list.
listdir() method in python is used to get the list of all files and directories in the specified directory.

**Program:**
import cv2
import os
path = 'E:\IP1'
imgs = []

files = os.listdir(path)
for file in files:
    filepath=path+"\\"+file
    imgs.append(cv2.imread(filepath))
i=0
im = []
for im in imgs:
    #cv2.imshow(files[i],imgs[i])
    im+=imgs[i]
    i=i+1
cv2.imshow("sum of four pictures",im)
meanImg = im/len(files)
cv2.imshow("mean of four pictures",meanImg)
cv2.waitKey(0)

**Output**
![image](https://user-images.githubusercontent.com/72294293/104426618-2fd09e80-55a8-11eb-88f2-d45eeb59614b.png)

**Q4.Write a program to convert color image into gray scale and binary image**

**Description**
Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.
A binary image is a monochromatic image that consists of pixels that can have one of exactly two colors, usually black and white.
cv2.threshold works as, if pixel value is greater than a threshold value, it is assigned one value (may be white), else it is assigned another value (may be black). 
destroyAllWindows() simply destroys all the windows we created. 
To destroy any specific window, use the function cv2. destroyWindow() where you pass the exact window name.

**Program:**
import cv2
img = cv2.imread("Dog.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("Binary Image",gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow("Binary Image",bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()













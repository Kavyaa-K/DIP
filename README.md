# DIP
Q1. Develop a program to display grayscale image using read and write the operation.

**Description**
imread() : is used for reading an image. 
imwrite(): is used to write an image in memory to disk. 
imshow() :to display an image. 
waitKey(): The function waits for specified milliseconds for any keyboard event.
destroyAllWindows():function to close all the windows. cv2. cvtColor() method is used to convert an image from one color space to another For color conversion, we use the function cv2. cvtColor(input_image, flag) where flag determines the type of conversion. For BGR Gray conversion we use the flags cv2.COLOR_BGR2GRAY

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





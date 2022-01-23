import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
img = cv.imread('2b827.png',0)
img = cv.medianBlur(img,5)
gry = img
(h, w) = gry.shape[:2]
gry = cv.resize(img, (w*2, h*2))
kernel = np.ones((2,2),np.uint8)
dilated_img = cv.dilate(img,kernel,iterations = 1)
cls = cv.morphologyEx(dilated_img, cv.MORPH_OPEN, None)

ret,th1 = cv.threshold(cls,127,255,cv.THRESH_BINARY)


th2 = cv.adaptiveThreshold(cls,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)

th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
ret,th4=cv.threshold(cls,127,255,cv.THRESH_OTSU)



txt = pytesseract.image_to_string(th1,lang='eng')
print("Nakon binarnog thresholding-a: ",txt,"\n")

txt = pytesseract.image_to_string(th2)
txt
print("Nakon Adaptive Mean Thresholding-a: ",txt,"\n")

txt = pytesseract.image_to_string(th3)
print("Nakon Adaptive Gaussian Thresholding-a: ",txt,"\n")
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Mean Gaussian']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
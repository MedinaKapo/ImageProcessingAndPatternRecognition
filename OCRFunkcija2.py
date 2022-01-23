import matplotlib.pyplot as plt
import matplotlib.image as img
import os
import csv
import numpy as np
import pandas as pd
import pytesseract
from tensorflow import keras as k
from keras.utils import np_utils
import cv2
from keras.models import  load_model
from PIL import Image
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, MaxPooling2D, Flatten, Conv2D, Dropout, Dense
from labeleCNN import info
import pytesseract as p
def ocrFunkcija2(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    image= cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
    plt.imshow(image)
    plt.show()
    kernel = np.ones((4,4), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    plt.imshow(image)
    plt.show()
    kernel = np.ones((4,4), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    plt.imshow(image)
    plt.show()
    image = cv2.GaussianBlur(image, (5,5), 0)
    plt.imshow(image)
    plt.show()
    ocr_result = p.image_to_string(image,lang="eng")
    print("Tekst: ", "\n")
    print(ocr_result)


ocrFunkcija2('2b827.png')
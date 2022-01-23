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

MODEL_FILENAME = "POOSProjekatOCR.h5"
model = load_model(MODEL_FILENAME)

def ocrfunkcija(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    x = [image[10:50, 30:50], image[10:50, 50:70], image[10:50, 70:90],
         image[10:50, 90:110], image[10:50, 110:130]]

    X_pred = []
    for i in range(5):
        X_pred.append(img_to_array(Image.fromarray(x[i])))

    X_pred = np.array(X_pred)
    X_pred /= 255.0

    y_pred = model.predict(X_pred)
    y_pred = np.argmax(y_pred, axis=1)
    print('Pretpostavka: ', end='')
    for res in y_pred:
        print(info[res], end='')

    print('\nTačan odgovor:    ', img_path[len(img_path) - 9:len(img_path) - 4])

#for filenames in os.listdir('val'):
   # path = 'val/' + filenames
    #print(path)
    #writer = csv.writer(f)
    #writer.writerow(filenames)

    #PREDPROCESIRANJE SLIKE
    #image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #ocrfunkcija(path)

ocrfunkcija('val/2enf4.png')


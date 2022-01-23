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
from PIL import Image
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, MaxPooling2D, Flatten, Conv2D, Dropout, Dense

y = []
#f = open('proba2.csv', 'w')
# Since there are subfolders inside the input directory, we've used nested loops
for filenames in os.listdir('train'):
    path = 'train/' + filenames

    # Labele su ustvari tacni odgovori
    for i in range(5):
        y.append(path[len(path) - 9:len(path) - 4][i])

y = np.array(y)

labelEnc = LabelEncoder().fit_transform(y)
oneHotEnc = OneHotEncoder(sparse = False).fit_transform(labelEnc.reshape(len(labelEnc),1))

info = {labelEnc[i] : y[i] for i in range(len(y))}

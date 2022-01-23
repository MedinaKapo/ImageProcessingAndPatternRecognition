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

X = []
y = []
#f = open('proba2.csv', 'w')
for filenames in os.listdir('train'):
    path = 'train/' + filenames
    #print(path)
    #writer = csv.writer(f)
    #writer.writerow(filenames)

    #PREDPROCESIRANJE SLIKE
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Podjela slike na segmente
    x = [image[10:50, 30:50], image[10:50, 50:70],
         image[10:50, 70:90], image[10:50, 90:110], image[10:50, 110:130]]

    # Labele su ustvari tacni odgovori
    for i in range(5):
        X.append(img_to_array(Image.fromarray(x[i])))
        y.append(path[len(path) - 9:len(path) - 4][i])

X = np.array(X)
y = np.array(y)


X=X.astype('float32')
X/=255

labelEnc = LabelEncoder().fit_transform(y)
oneHotEnc = OneHotEncoder(sparse = False).fit_transform(labelEnc.reshape(len(labelEnc),1))

X_train, X_test, y_train, y_test = train_test_split(X, oneHotEnc, test_size = 0.2, random_state = 42)

row, col = X.shape[1],X.shape[2]
categories = oneHotEnc.shape[1]

info = {labelEnc[i] : y[i] for i in range(len(y))}


model = Sequential()

model.add(Conv2D(filters=16,kernel_size=(3,3), padding='same', input_shape=(row,col,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(filters=16,kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(filters=32,kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(filters=32,kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(1500))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(categories))
model.add(Activation("softmax"))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop' ,
              metrics = ['accuracy'])

print(model.summary())

batch_size = 64
epochs = 200

history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test),
          shuffle=True)

scores = model.evaluate(X_test, y_test, verbose=1)
#print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

model.save('POOSProjekatOCR.h5')


def ocrfunkcija(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

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

#ocrfunkcija('2nf26.png')


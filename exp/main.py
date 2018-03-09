"""
Simple CNN for RGB+D face recognition. For now it has ~65% categorical accuracy.


Assumes existence of folder 'files' in the same directory. Folder 'files' must
be a copy (or link) of www.vap.aau.dk/files from the shared database.

TODO (tomek): test this code (after addition of superface_dataset), evaluate the model
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, AveragePooling2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.applications import ResNet50
import face_recognition
import matplotlib.pyplot as plt
import numpy as np
from array import array
import sys
import os
from skimage.filters.rank import entropy
from skimage.morphology import disk

import tools
from tools import image_size, TYPES
from db_helper import DBHelper

sys.setrecursionlimit(1000000)

# TODO(tomek): fix folder subject010 and change 19->12
TOTAL_SUBJECTS_COUNT = 19 + 26 + 31

def display_photo(path):
    FORMATS = ['PHDE', 'PHIR']
    photo = np.asarray(data_arr)
    photo = photo.reshape(height, width)
    plt.gray()
    plt.imshow(photo)
    plt.show()

print("Loading data..")

# Load data
x_train = []
y_train = []
x_test = []
y_test = []

def load_database(db_name, train_split=2/3):
    # train_split is a fraction of data used for training. e.g. train_split=2/3
    # means that out of 3 samples 2 are used for training and 1 for test
    db_helper = DBHelper(db_name)
    print('Loading database %s' % db_name)
    for i in range(db_helper.subjects_count):
        print('subject', i)
        l = int(db_helper.imgs_per_subject(i) * train_split) # TRAIN/TEST split
        for j in range(db_helper.imgs_per_subject(i)):
            print(i, j)
            x = db_helper.build_input_vector(i, j)
            y = i + 1
            if x is None or y is None:
                continue
            if j > l:
                x_test.append(x)
                y_test.append(y)
            else:
                x_train.append(x)
                y_train.append(y)

load_database('superface_dataset')
load_database('ias_lab_rgbd')
load_database('www.vap.aau.dk')

# Reshape input
TRAIN_SIZE = len(x_train)
TEST_SIZE = len(x_test)
X_train = np.zeros((TRAIN_SIZE, TYPES * image_size, image_size, 1))
Y_train = np.zeros((TRAIN_SIZE, TOTAL_SUBJECTS_COUNT))
X_test = np.zeros((TEST_SIZE, TYPES * image_size, image_size, 1))
Y_test = np.zeros((TEST_SIZE, TOTAL_SUBJECTS_COUNT))
for i in range(TRAIN_SIZE):
    X_train[i] = x_train[i].reshape((TYPES * image_size, image_size, 1))
    Y_train[i,y_train[i]-1] = 1
for i in range(TEST_SIZE):
    X_test[i] = x_test[i].reshape((TYPES * image_size, image_size, 1))
    Y_test[i,y_test[i]-1] = 1
x_train = []
y_train = []
x_test = []
y_test = []

print("Loaded data")
# If you want, display the first input image. It is already normalized to [0;1]
tools.show_image(X_train[0].reshape((TYPES * image_size, image_size)));

model = Sequential()
model.add(Conv2D(20,
                 kernel_size=(6,6),
                 activation='relu',
                 input_shape=(TYPES * image_size, image_size, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(20,
                 kernel_size=(6,6),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(20,
                 kernel_size=(6,6),
                 activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(TOTAL_SUBJECTS_COUNT, activation='softmax'))

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

train_generator = ImageDataGenerator().flow(X_train, Y_train)
test_generator = ImageDataGenerator().flow(X_test, Y_test)

# 50 epochs is obviously not enough, repeat until convergence manually.
# NOTE: X_test and Y_test are passed here as validation data so that score will
# be displayed live while learning
model.fit_generator(train_generator,
          validation_data=test_generator,
          epochs=50,
          steps_per_epoch=10,
          verbose=True,
          validation_steps=1)


loss = model.evaluate(X_test, Y_test)
print(loss)

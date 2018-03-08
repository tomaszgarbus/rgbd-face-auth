"""
Main.py without entropy maps.

Assumes existence of folder 'files' in the same directory. Folder 'files' must
be a copy (or link) of www.vap.aau.dk/files from the shared database.
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
import tools
from random import shuffle
from skimage.filters.rank import entropy
from skimage.morphology import disk


TRAIN_PATH='files'
SUBJECTS_COUNT=31
image_size=50
MARGIN=10
TYPES = 2

sys.setrecursionlimit(1000000)

def display_photo(path):
    FORMATS = ['PHDE', 'PHIR']
    photo = np.asarray(data_arr)
    photo = photo.reshape(height, width)
    plt.gray()
    plt.imshow(photo)
    plt.show()

def load_depth_photo(path):
    with open(path, 'rb') as f:
        format_arr = np.fromfile(f, dtype='i1', count=4)
        assert ''.join(map(chr, format_arr)) == 'PHDE'
        size_arr = np.fromfile(f, dtype='i4', count=2)
        width, height = size_arr
        data_arr = np.fromfile(f, dtype='f', count=height * width)
        photo = np.asarray(data_arr)
        photo = photo.reshape(height, width)
        return photo

def load_train_subject(subject_no, im_no):
    x = []
    y = []
    for i in range(1,18):
        # Load depth and color photo
        path_depth = TRAIN_PATH + '/%d/0%02d_%d_d.depth' % (subject_no, i, im_no)
        path_color = TRAIN_PATH + '/%d/0%02d_%d_c.png' % (subject_no, i, im_no)
        color_photo = tools.load_color_image_from_file(path_color)
        depth_photo = load_depth_photo(path_depth)
        # Resize to common size
        color_photo = tools.change_image_mode('RGBA', 'RGB', color_photo)
        color_photo = tools.rgb_image_resize(color_photo, (depth_photo.shape[1], depth_photo.shape[0]))
        # Locate face
        face_coords = face_recognition.face_locations(color_photo)
        # Find features of the face
        if len(face_coords) == 1:
            (x1,y1,x2,y2) = face_coords[0]
            depth_face = depth_photo[x1:x2,y2:y1]
            color_face = color_photo[x1:x2,y2:y1]
            #tools.show_image(color_face)
            landmarks = face_recognition.face_landmarks(color_photo)[0]
            depth_face = tools.gray_image_resize(depth_face, (image_size, image_size))
            depth_face = depth_face/np.max(depth_face)
            color_face = tools.rgb_image_resize(color_face, (image_size, image_size))
            tmp = np.zeros((TYPES * image_size, image_size))
            grey_face = tools.change_image_mode('RGB', 'L', color_face)
            grey_face = grey_face/np.max(grey_face)
            entr_grey_face = entropy(grey_face, disk(5))
            entr_grey_face = entr_grey_face/np.max(entr_grey_face)
            entr_depth_face = entropy(depth_face, disk(5))
            entr_depth_face = entr_depth_face/np.max(entr_depth_face)
            tmp[0:image_size] = depth_face
            tmp[image_size:image_size*2] = grey_face
            tmp[image_size*2:image_size*3] = entr_grey_face
            tmp[image_size*3:image_size*4] = entr_depth_face
            x.append(tmp)
            y.append(subject_no)
    return x, y

print("Loading data..")

# Load data
x_train = []
y_train = []
x_test = []
y_test = []
for i in range(1, SUBJECTS_COUNT+1):
    print('subject', i)
    for j in range(1, 4):
        print(i, j)
        (x,y) = load_train_subject(i, j)
        l = (len(x) * 2) // 3
        x_train += x[:l]
        y_train += y[:l]
        x_test += x[l:]
        y_test += y[l:]

# Split data

TRAIN_SIZE = len(x_train)
TEST_SIZE = len(x_test)
X_train = np.zeros((TRAIN_SIZE, TYPES * image_size, image_size, 1))
Y_train = np.zeros((TRAIN_SIZE, SUBJECTS_COUNT))
X_test = np.zeros((TEST_SIZE, TYPES * image_size, image_size, 1))
Y_test = np.zeros((TEST_SIZE, SUBJECTS_COUNT))
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
#tools.show_image(X_train[0].reshape((TYPES * image_size, image_size)));

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
model.add(Dense(SUBJECTS_COUNT, activation='softmax'))

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

train_generator = ImageDataGenerator().flow(X_train, Y_train)

model.fit_generator(train_generator,
          epochs=1000,
          steps_per_epoch=10,
          verbose=True,
          validation_steps=1)


loss = model.evaluate(X_test, Y_test)
print(loss)

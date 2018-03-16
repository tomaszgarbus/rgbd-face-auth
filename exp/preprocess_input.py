"""
Preprocessing the input images is very expensive, as we want to crop them to
faces and calculate entropy and (TODO) saliency maps.
Use this script to generate (X|Y)_(train|test).npy files and load them directly
in main.py.
"""

from db_helper import DBHelper, SUBJECTS_COUNTS
import numpy as np
from tools import image_size, TYPES
from skimage.filters.rank import entropy
from skimage.morphology import disk

TOTAL_SUBJECTS_COUNT = SUBJECTS_COUNTS['superface_dataset'] + SUBJECTS_COUNTS['www.vap.aau.dk'] + SUBJECTS_COUNTS['ias_lab_rgbd']

# Load data
x_train = []
y_train = []
x_test = []
y_test = []


def build_input_vector(greyd_face):
    """ Concatenates: grey_face, depth_face, entr_grey_face, entr_depth_face"""
    (grey_face, depth_face) = greyd_face
    if grey_face is None or depth_face is None:
        return None
    tmp = np.zeros((TYPES * image_size, image_size))
    entr_grey_face = entropy(grey_face, disk(5))
    entr_grey_face = entr_grey_face / np.max(entr_grey_face)
    entr_depth_face = entropy(depth_face, disk(5))
    entr_depth_face = entr_depth_face / np.max(entr_depth_face)
    tmp[0:image_size] = depth_face
    tmp[image_size:image_size * 2] = grey_face
    tmp[image_size * 2:image_size * 3] = entr_grey_face
    tmp[image_size * 3:image_size * 4] = entr_depth_face
    return tmp

def load_database(db_name, offset, train_split=2/3):
    # train_split is a fraction of data used for training. e.g. train_split=2/3
    # means that out of 3 samples 2 are used for training and 1 for test
    db_helper = DBHelper(db_name)
    print('Loading database %s' % db_name)
    for i in range(db_helper.subjects_count):
        print('subject', i)
        l = int(db_helper.imgs_per_subject(i) * train_split) # TRAIN/TEST split
        for j in range(db_helper.imgs_per_subject(i)):
            print(i, j)
            x = build_input_vector(db_helper.load_greyd_face(i, j))
            y = offset + i + 1
            if x is None or y is None:
                continue
            if j > l:
                x_test.append(x)
                y_test.append(y)
            else:
                x_train.append(x)
                y_train.append(y)


if __name__ == '__main__':
    load_database('superface_dataset', 0)
    load_database('ias_lab_rgbd', SUBJECTS_COUNTS['superface_dataset'])
    load_database('www.vap.aau.dk', SUBJECTS_COUNTS['superface_dataset'] + SUBJECTS_COUNTS['ias_lab_rgbd'])

    # Reshape input
    TRAIN_SIZE = len(x_train)
    TEST_SIZE = len(x_test)
    X_train = np.zeros((TRAIN_SIZE, TYPES * image_size, image_size, 1))
    Y_train = np.zeros((TRAIN_SIZE, TOTAL_SUBJECTS_COUNT))
    X_test = np.zeros((TEST_SIZE, TYPES * image_size, image_size, 1))
    Y_test = np.zeros((TEST_SIZE, TOTAL_SUBJECTS_COUNT))
    for i in range(TRAIN_SIZE):
        X_train[i] = x_train[i].reshape((TYPES * image_size, image_size, 1))
        Y_train[i, y_train[i]-1] = 1
    for i in range(TEST_SIZE):
        X_test[i] = x_test[i].reshape((TYPES * image_size, image_size, 1))
        Y_test[i, y_test[i]-1] = 1
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(len(X_train)):
        if np.isnan(X_train[i]).any():
            # TODO: do something smarter - investigate how nan values sneaked into
            # input data. This has probably happened when computing entropy maps
            # (I think I saw a warning from numpy).
            # The invalid images were: 2871, 2873, 2874
            X_train[i] = X_train[0]
            Y_train[i] = Y_train[0]

    np.save('X_train', X_train)
    np.save('Y_train', Y_train)
    np.save('X_test', X_test)
    np.save('Y_test', Y_test)

"""
Preprocessing the input images is very expensive, as we want to crop them to
faces and calculate entropy and (TODO) HOGs of entropy maps.
Use this script to generate (X|Y)_(train|test).npy files and load them directly
in main.py.
"""

from common.db_helper import DBHelper, Database, DB_LOCATION
import numpy as np
from common import tools
from common.tools import IMG_SIZE
from skimage.filters.rank import entropy
from skimage.morphology import disk
from face_rotation.rotate import rotate_greyd_img_by_angle, preprocess_images
import os

def build_input_vector(greyd_face):
    """ Concatenates: grey_face, depth_face, entr_grey_face, entr_depth_face"""
    (grey_face, depth_face) = greyd_face
    if grey_face is None or depth_face is None:
        return None
    tmp = np.zeros((4 * IMG_SIZE, IMG_SIZE))
    entr_grey_face = entropy(grey_face, disk(5))
    entr_grey_face = entr_grey_face / np.max(entr_grey_face)
    entr_depth_face = entropy(depth_face, disk(5))
    entr_depth_face = entr_depth_face / np.max(entr_depth_face)
    tmp[0:IMG_SIZE] = depth_face
    tmp[IMG_SIZE:IMG_SIZE * 2] = grey_face
    tmp[IMG_SIZE * 2:IMG_SIZE * 3] = entr_grey_face
    tmp[IMG_SIZE * 3:IMG_SIZE * 4] = entr_depth_face
    return tmp

total_rotated = 0

def load_database(database, offset, override_test_set=False):
    global total_rotated
    total_rotated_db = 0
    print('Loading database %s' % database.get_name())
    for i in range(database.subjects_count()):
        print('Subject', i)
        for j in range(database.imgs_per_subject(i)):
            print('Photo %d/%d' % (j, database.imgs_per_subject(i)))
            greyd_face = database.load_greyd_face(i, j)
            x = build_input_vector(greyd_face)
            y = offset + i + 1
            if x is None or y is None:
                continue
            if database.is_photo_in_test_set(i, j):
                x_test.append(x)
                y_test.append(y)
            else:
                if database.is_photo_frontal(i, j):
                    for theta_x in np.linspace(-0.2, 0.2, 3):
                        for theta_y in np.linspace(-0.2, 0.2, 3):
                            img_grey = np.copy(greyd_face[0])
                            img_depth = np.copy(greyd_face[1])
                            rotated_greyd_face = rotate_greyd_img_by_angle((img_grey, img_depth), theta_x, theta_y)
                            x = build_input_vector(rotated_greyd_face)
                            x_train.append(x)
                            y_train.append(y)
                            total_rotated += 1
                            total_rotated_db += 1
                else:
                    x_train.append(x)
                    y_train.append(y)
    print("Total rotated ", total_rotated, total_rotated_db)


if __name__ == '__main__':
    # Load data
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    helper = DBHelper()
    TOTAL_SUBJECTS_COUNT = helper.all_subjects_count()
    print(TOTAL_SUBJECTS_COUNT)

    sum_offset = 0
    for database in helper.get_databases():
        load_database(database, sum_offset)
        sum_offset += database.subjects_count()

    # Reshape input
    TRAIN_SIZE = len(x_train)
    TEST_SIZE = len(x_test)
    X_train = np.zeros((TRAIN_SIZE, 4 * IMG_SIZE, IMG_SIZE, 1))
    Y_train = np.zeros((TRAIN_SIZE, TOTAL_SUBJECTS_COUNT))
    X_test = np.zeros((TEST_SIZE, 4 * IMG_SIZE, IMG_SIZE, 1))
    Y_test = np.zeros((TEST_SIZE, TOTAL_SUBJECTS_COUNT))
    for i in range(TRAIN_SIZE):
        X_train[i] = x_train[i].reshape((4 * IMG_SIZE, IMG_SIZE, 1))
        Y_train[i, y_train[i]-1] = 1
    for i in range(TEST_SIZE):
        X_test[i] = x_test[i].reshape((4 * IMG_SIZE, IMG_SIZE, 1))
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

    assert os.path.isdir(DB_LOCATION), "database directory not found"
    if not os.path.isdir(DB_LOCATION + '/gen'):
        os.makedirs(DB_LOCATION + '/gen')

    np.save(DB_LOCATION + '/gen/rotation_on_train_set_X_train', X_train)
    np.save(DB_LOCATION + '/gen/rotation_on_train_set_Y_train', Y_train)
    np.save(DB_LOCATION + '/gen/rotation_on_train_set_X_test', X_test)
    np.save(DB_LOCATION + '/gen/rotation_on_train_set_Y_test', Y_test)

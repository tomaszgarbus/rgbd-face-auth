"""
    Preprocessing the input images is very expensive, as we want to crop them to
    faces and calculate entropy and HOGs of entropy maps.
    Use this script to generate (X|Y)_(train|test).npy files and load them directly
    in main.py.
"""

import numpy as np
import os
import logging
from progress.bar import Bar

from common.constants import NUM_CLASSES
from common.db_helper import Database, DBHelper, DB_LOCATION
from experiments.no_rotation_channels_without_hogs.constants import EXP_NAME, NN_INPUT_SIZE
from controller.normalization import normalized, hog_and_entropy


def build_input_vector(face):
    """ Concatenates: grey_face, depth_face, entr_grey_face, entr_depth_face"""
    (grey_face, depth_face) = (face.grey_img, face.depth_img)
    if grey_face is None or depth_face is None:
        return None
    if np.isnan(grey_face).any() or np.isnan(depth_face).any():
        return None
    try:
        face = normalized(face, rotate=False)
        face = hog_and_entropy(face)
    except ValueError:
        return None
    return face.get_channels()[:, :, :4]


def load_database(database: Database, offset: int):
    logging.info('Loading database %s' % database.get_name())
    for i in range(database.subjects_count()):
        logging.info('Subject {0}/{1}'.format(i, database.subjects_count()))
        bar = Bar('Subject ' + str(i), max=database.imgs_per_subject(i))
        for j in range(database.imgs_per_subject(i)):
            face = database.load_greyd_face(i, j)
            x = build_input_vector(face)
            y = offset + i + 1
            if x is None or y is None:
                bar.next()
                continue
            # tools.show_image(x)
            if database.is_photo_in_test_set(i, j):
                x_test.append(x)
                y_test.append(y)
                # tools.show_image(x)
            else:
                x_train.append(x)
                y_train.append(y)

            bar.next()
        bar.finish()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Load data
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    helper = DBHelper()

    sum_offset = 0
    for database in helper.get_databases():
        if database.get_name() not in ['www.vap.aau.dk', 'superface_dataset']:
            load_database(database, sum_offset)
            sum_offset += database.subjects_count()

    # Reshape input
    TRAIN_SIZE = len(x_train)
    TEST_SIZE = len(x_test)
    X_train = np.zeros((TRAIN_SIZE, NN_INPUT_SIZE[0], NN_INPUT_SIZE[1], NN_INPUT_SIZE[2]))
    Y_train = np.zeros((TRAIN_SIZE, NUM_CLASSES))
    X_test = np.zeros((TEST_SIZE, NN_INPUT_SIZE[0], NN_INPUT_SIZE[1], NN_INPUT_SIZE[2]))
    Y_test = np.zeros((TEST_SIZE, NUM_CLASSES))
    for i in range(TRAIN_SIZE):
        X_train[i] = x_train[i].reshape((NN_INPUT_SIZE[0], NN_INPUT_SIZE[1], NN_INPUT_SIZE[2]))
        Y_train[i, y_train[i]-1] = 1
    for i in range(TEST_SIZE):
        X_test[i] = x_test[i].reshape((NN_INPUT_SIZE[0], NN_INPUT_SIZE[1], NN_INPUT_SIZE[2]))
        Y_test[i, y_test[i]-1] = 1
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(len(X_train)):
        if np.isnan(X_train[i]).any():
            # TODO: check if nan values are still an issue
            X_train[i] = X_train[0]
            Y_train[i] = Y_train[0]

    assert os.path.isdir(DB_LOCATION), "database directory not found"
    if not os.path.isdir(DB_LOCATION + '/gen'):
        os.makedirs(DB_LOCATION + '/gen')

    np.save(DB_LOCATION + '/gen/' + EXP_NAME + '_X_train', X_train)
    np.save(DB_LOCATION + '/gen/' + EXP_NAME + '_Y_train', Y_train)
    np.save(DB_LOCATION + '/gen/' + EXP_NAME + '_X_test', X_test)
    np.save(DB_LOCATION + '/gen/' + EXP_NAME + '_Y_test', Y_test)

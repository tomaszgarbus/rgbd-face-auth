"""
Preprocessing input for kNN test - find, rotate and recentre face.
"""

from common.db_helper import DBHelper, Database, DB_LOCATION
import numpy as np
from common import tools
from common.tools import IMG_SIZE
from face_rotation import rotate
from face_rotation.find_angle import find_angle
from face_rotation.recentre import recentre
import os

MARGIN = 14
SIZE = IMG_SIZE - 2 * MARGIN

def build_input_vector(greyd_face):
    """ rotates and recentres face. Also cuts out some margin """
    (grey_face, depth_face) = greyd_face
    if grey_face is None or depth_face is None:
        return None
    rotate.preprocess_images(depth_face, grey_face)
    theta_x, theta_y, theta_z, center = find_angle(grey_face, depth_face)

    # Apply rotation
    rotated_grey, rotated_depth = rotate.rotate_greyd_img((grey_face, depth_face),
                                                          theta_x=theta_x,
                                                          theta_y=theta_y,
                                                          theta_z=theta_z)

    theta_x, theta_y, theta_z, center = find_angle(rotated_grey, rotated_depth)
    rotated_grey, rotated_depth = recentre(rotated_grey, rotated_depth, center)
    rotated_grey = rotated_grey[MARGIN:-MARGIN, MARGIN:-MARGIN]
    rotated_depth = rotated_depth[MARGIN:-MARGIN, MARGIN:-MARGIN]
    tmp = np.zeros((2 * SIZE, SIZE))
    tmp[0:SIZE] = rotated_grey
    tmp[SIZE:SIZE * 2] = rotated_depth
    #tools.show_image(tmp)
    return tmp

def load_database(database, offset, override_test_set=False):
    print('Loading database %s' % database.get_name())
    for i in range(database.subjects_count()):
        print('Subject', i)
        for j in range(database.imgs_per_subject(i)):
            print('Photo %d/%d' % (j, database.imgs_per_subject(i)))
            x = build_input_vector(database.load_greyd_face(i, j))
            y = offset + i + 1
            if x is None or y is None:
                continue
            if database.is_photo_in_test_set(i, j):
                x_test.append(x)
                y_test.append(y)
            else:
                x_train.append(x)
                y_train.append(y)


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
    X_train = np.zeros((TRAIN_SIZE, 2 * SIZE, SIZE, 1))
    Y_train = np.zeros((TRAIN_SIZE, TOTAL_SUBJECTS_COUNT))
    X_test = np.zeros((TEST_SIZE, 2 * SIZE, SIZE, 1))
    Y_test = np.zeros((TEST_SIZE, TOTAL_SUBJECTS_COUNT))
    for i in range(TRAIN_SIZE):
        X_train[i] = x_train[i].reshape((2 * SIZE, SIZE, 1))
        Y_train[i, y_train[i]-1] = 1
    for i in range(TEST_SIZE):
        X_test[i] = x_test[i].reshape((2 * SIZE, SIZE, 1))
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

    np.save(DB_LOCATION + '/gen/rotated_knn_X_train', X_train)
    np.save(DB_LOCATION + '/gen/rotated_knn_Y_train', Y_train)
    np.save(DB_LOCATION + '/gen/rotated_knn_X_test', X_test)
    np.save(DB_LOCATION + '/gen/rotated_knn_Y_test', Y_test)

"""
    Base input preprocessor for neural network.
"""

import numpy as np
import os
import logging
from progress.bar import Bar
from typing import Callable, Tuple

from model.face import Face
from common.constants import NUM_CLASSES
from common.db_helper import Database, DBHelper, DB_LOCATION


class InputPreprocessor:
    def __init__(self,
                 exp_name: str,
                 nn_input_size: Tuple[int, int, int],
                 build_input_vector):
        self.exp_name = exp_name
        self.nn_input_size = nn_input_size
        self.build_input_vector = build_input_vector
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

    def load_database(self, database: Database, offset: int):
        logging.info('Loading database %s' % database.get_name())
        for i in range(database.subjects_count()):
            logging.info('Subject {0}/{1}'.format(i, database.subjects_count()))
            bar = Bar('Subject ' + str(i), max=database.imgs_per_subject(i))
            for j in range(database.imgs_per_subject(i)):
                face = database.load_greyd_face(i, j)
                x = self.build_input_vector(face)
                y = offset + i + 1
                if x is None or y is None:
                    bar.next()
                    continue
                # tools.show_image(x)
                if database.is_photo_in_test_set(i, j):
                    self.x_test.append(x)
                    self.y_test.append(y)
                    # tools.show_image(x)
                else:
                    self.x_train.append(x)
                    self.y_train.append(y)
    
                bar.next()
            bar.finish()

    def preprocess(self):
        logging.basicConfig(level=logging.INFO)
    
        helper = DBHelper()
    
        sum_offset = 0
        for database in helper.get_databases():
            if database.get_name() not in ['www.vap.aau.dk', 'superface_dataset']:
                self.load_database(database, sum_offset)
                sum_offset += database.subjects_count()
    
        # Reshape input
        TRAIN_SIZE = len(self.x_train)
        TEST_SIZE = len(self.x_test)
        X_train = np.zeros((TRAIN_SIZE, self.nn_input_size[0], self.nn_input_size[1], self.nn_input_size[2]))
        Y_train = np.zeros((TRAIN_SIZE, NUM_CLASSES))
        X_test = np.zeros((TEST_SIZE, self.nn_input_size[0], self.nn_input_size[1], self.nn_input_size[2]))
        Y_test = np.zeros((TEST_SIZE, NUM_CLASSES))
        for i in range(TRAIN_SIZE):
            X_train[i] = self.x_train[i].reshape((self.nn_input_size[0], self.nn_input_size[1], self.nn_input_size[2]))
            Y_train[i, self.y_train[i]-1] = 1
        for i in range(TEST_SIZE):
            X_test[i] = self.x_test[i].reshape((self.nn_input_size[0], self.nn_input_size[1], self.nn_input_size[2]))
            Y_test[i, self.y_test[i]-1] = 1
    
        for i in range(len(X_train)):
            if np.isnan(X_train[i]).any():
                # TODO: check if nan values are still an issue
                X_train[i] = X_train[0]
                Y_train[i] = Y_train[0]
    
        assert os.path.isdir(DB_LOCATION), "database directory not found"
        if not os.path.isdir(DB_LOCATION + '/gen'):
            os.makedirs(DB_LOCATION + '/gen')
    
        np.save(DB_LOCATION + '/gen/' + self.exp_name + '_X_train', X_train)
        np.save(DB_LOCATION + '/gen/' + self.exp_name + '_Y_train', Y_train)
        np.save(DB_LOCATION + '/gen/' + self.exp_name + '_X_test', X_test)
        np.save(DB_LOCATION + '/gen/' + self.exp_name + '_Y_test', Y_test)

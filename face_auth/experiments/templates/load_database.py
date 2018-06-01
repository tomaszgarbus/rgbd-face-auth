
from common.db_helper import DB_LOCATION
from common.tools import show_image
import numpy as np


def load_data(self, range_beg: int = 0, range_end: int = 52) -> None:
    """
    :param range_beg, range_end: only samples such that label \in [range_beg, range_end) will be
        used. Sensible values for (range_beg, range_end) would be:
        * 00, 52 -> to use eurecom only
        * 52, 78 -> to use ias_lab_rgbd_only
        * 78, 98 -> to use superface_dataset only
    :return: self.(x|y)_(train|test) are set as a result
    """

    # Load stored numpy arrays from files.
    print("Loading data..")
    x_train = np.load(DB_LOCATION + '/gen/' + self.experiment_name + '_X_train.npy')
    y_train = np.load(DB_LOCATION + '/gen/' + self.experiment_name + '_Y_train.npy')
    x_test = np.load(DB_LOCATION + '/gen/' + self.experiment_name + '_X_test.npy')
    y_test = np.load(DB_LOCATION + '/gen/' + self.experiment_name + '_Y_test.npy')
    train_indices = []
    test_indices = []

    # Filter out samples out of [range_beg, range_end).
    for i in range(len(self.y_train)):
        if range_end > np.argmax(self.y_train[i]) >= range_beg:
            train_indices.append(i)
    for i in range(len(self.y_test)):
        if range_end > np.argmax(self.y_test[i]) >= range_beg:
            test_indices.append(i)
    self.x_train = self.x_train[train_indices]
    self.y_train = self.y_train[train_indices]
    self.x_test = self.x_test[test_indices]
    self.y_test = self.y_test[test_indices]
    # Show first input if you want
    show_image(self.x_train[0].reshape([self.input_shape[0], self.input_shape[1] * self.input_shape[2]]))

    # Image augmentation.
    if not self.augment_on_the_fly:
        self._augment_train_set()

    print("Loaded data..")
    return x_train
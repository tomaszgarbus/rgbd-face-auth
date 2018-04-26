"""
Simple CNN for RGB+D face recognition.
"""

import numpy as np
import logging
import tensorflow as tf
from math import sqrt
from random import sample
from progress.bar import Bar

from common.tools import show_image
from common.db_helper import DB_LOCATION
from common.constants import IMG_SIZE

NUM_CLASSES = 129


class NeuralNet:
    # TODO: optionally pass through constructor
    mb_size = 32
    conv_layers = [32, 32]
    kernel_size = [5, 5]
    dense_layers = [64, NUM_CLASSES]
    dropout = 0.5
    learning_rate = 0.2
    nb_epochs = 50000
    accs = []
    val_accs = []

    _confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

    def _get_data(self):
        logging.debug("Loading data..")
        self.x_train = np.load(DB_LOCATION + '/gen/face_rotation_X_train.npy')
        self.y_train = np.load(DB_LOCATION + '/gen/face_rotation_Y_train.npy')
        self.x_test = np.load(DB_LOCATION + '/gen/face_rotation_X_test.npy')
        self.y_test = np.load(DB_LOCATION + '/gen/face_rotation_Y_test.npy')
        logging.debug("Loaded data..")

    def _create_model(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[self.mb_size, 2 * IMG_SIZE, IMG_SIZE, 1])
        self.y = tf.placeholder(dtype=tf.float32, shape=[self.mb_size, NUM_CLASSES])

        signal = self.x
        for num_filters in self.conv_layers:
            signal = tf.layers.batch_normalization(signal)

            # Init weights with std.dev = sqrt(2 / N)
            input_size = int(signal.get_shape()[1]) * int(signal.get_shape()[2]) * int(signal.get_shape()[3])
            w_init = tf.initializers.random_normal(stddev=sqrt(2 / input_size))

            cur_conv_layer = tf.layers.conv2d(inputs=signal,
                                              filters=num_filters,
                                              kernel_size=self.kernel_size,
                                              kernel_initializer=w_init,
                                              padding='same')
            cur_pool_layer = tf.layers.max_pooling2d(inputs=cur_conv_layer,
                                                     pool_size=[2, 2],
                                                     strides=2)

            signal = cur_pool_layer

        signal = tf.reshape(signal, [self.mb_size, -1])

        for num_neurons in self.dense_layers[:-1]:
            signal = tf.layers.batch_normalization(signal)

            cur_dropout_layer = tf.layers.dropout(inputs=signal,
                                                  rate=self.dropout)

            signal = cur_dropout_layer

            # Init weights with std.dev = sqrt(2 / N)
            input_size = int(signal.get_shape()[1])
            w_init = tf.initializers.random_normal(stddev=sqrt(2 / input_size))

            cur_dense_layer = tf.layers.dense(inputs=signal,
                                              units=num_neurons,
                                              activation=tf.nn.sigmoid,
                                              kernel_initializer=w_init)

            signal = cur_dense_layer

        # Apply last dense layer
        cur_layer = tf.layers.dense(inputs=signal,
                                    units=self.dense_layers[-1])
        signal = cur_layer

        self.preds = tf.argmax(self.y, axis=1)
        self.loss = tf.losses.mean_squared_error(self.y, signal)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.y, axis=1), tf.argmax(signal, axis=1)), tf.float32))
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def __init__(self):
        self._get_data()

        # Initialize logging.
        self.logger = logging.Logger("main_logger", level=logging.INFO)
        log_file = 'log.txt'
        formatter = logging.Formatter(
            fmt='{levelname:<7} {message}',
            style='{'
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def train_on_batch(self, batch_x, batch_y):
        """
        :return: [loss, accuracy]
        """
        results = self.sess.run([self.loss, self.accuracy, self.preds, self.train_op],
                                 feed_dict={self.x: batch_x, self.y: batch_y})
        self.accs.append(results[1])
        return results[:2]

    def test_on_batch(self, batch_x, batch_y):
        """
        Note that this function does not fetch |self.train_op|, so that the weights
        are not updated.
        :param batch_x:
        :param batch_y:
        :return: [loss, accuracy]
        """
        results = self.sess.run([self.loss, self.accuracy, self.preds],
                                feed_dict={self.x: batch_x, self.y: batch_y})
        self.val_accs.append(results[1])
        preds = results[2]
        # Update confusion matrix
        for i in range(self.mb_size):
            self._confusion_matrix[np.argmax(batch_y[i]),preds[i]] += 1.
        return results[:2]

    def train_and_evaluate(self) -> None:
        """
        Train and evaluate model.
        """
        with tf.Session() as self.sess:
            # Initialize computation graph.
            self._create_model()

            # Initialize variables.
            tf.global_variables_initializer().run()

            bar = Bar('', max=100, suffix='%(index)d/%(max)d ETA: %(eta)ds')
            for epoch_no in range(self.nb_epochs):
                # Train model on next batch

                batch = sample(list(range(self.x_train.shape[1])), self.mb_size)
                batch_x, batch_y = self.x_train[batch], self.y_train[batch]
                results = self.train_on_batch(batch_x, batch_y)

                bar.message = 'loss: {0[0]:.8f} acc: {0[1]:.3f} mean_acc: {1:.3f}'.format(results, np.mean(self.accs))

                if epoch_no % 100 == 0:
                    bar.finish()
                    bar = Bar('', max=100, suffix='%(index)d/%(max)d ETA: %(eta)ds')
                    #self.logger.info("Epoch {0}: Loss: {1[0]}, accuracy: {1[1]}, mean acc: {2}".format(epoch_no, results, np.mean(self.accs)))
                    batch_t = sample(list(range(self.x_test.shape[1])), self.mb_size)
                    batch_x_t, batch_y_t = self.x_test[batch], self.y_test[batch]
                    test_results = self.test_on_batch(batch_x_t, batch_y_t)
                    self.logger.info("(Test(batch):   Loss: {0[0]}, accuracy: {0[1]}, mean acc: {1}".format(test_results, np.mean(self.val_accs)))
                bar.next()



if __name__ == '__main__':
    net = NeuralNet()
    net.train_and_evaluate()
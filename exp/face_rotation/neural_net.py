"""
Simple CNN for RGB+D face recognition.
"""

import numpy as np
import logging
import tensorflow as tf
from math import sqrt
from random import sample
from typing import Optional, List, Tuple
from progress.bar import Bar
from imgaug import augmenters as ia

from common.db_helper import DB_LOCATION
from common.constants import NN_INPUT_SIZE
from common.tools import show_image

NUM_CLASSES = 129


class NeuralNet:
    # TODO: try initializing convolution filters with Gabor filters instead of random

    # Mini batch size
    mb_size = 32

    # Number of filters in each convolutional layer
    filters_count = [32, 64]

    # Size of kernel, common for each convolutional layer
    kernel_size = [5, 5]

    # Neurons count in each dense layer
    dense_layers = [32, NUM_CLASSES]

    # Dropout after each dense layer (excluding last)
    dropout = 0.5

    learning_rate = 0.2
    nb_epochs = 50000

    # History of accuracies on train set
    accs = []

    # History of accuracies on test set
    val_accs = []

    _confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

    def __init__(self,
                 mb_size: Optional[int] = None,
                 filters_count: Optional[List[int]] = None,
                 kernel_size: Optional[Tuple[int, int]] = None,
                 dense_layers: Optional[List[int]] = None,
                 learning_rate: Optional[float] = None,
                 nb_epochs: Optional[int] = None,
                 dropout_rate: Optional[int] = None,
                 min_label: int = 0,
                 max_label: int = NUM_CLASSES
                 ):
        if mb_size is not None:
            self.mb_size = mb_size
        if filters_count is not None:
            self.filters_count = filters_count
        if kernel_size is not None:
            self.kernel_size = kernel_size
        if dense_layers is not None:
            self.dense_layers = dense_layers
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if nb_epochs is not None:
            self.nb_epochs = nb_epochs
        if dropout_rate is not None:
            self.dropout = dropout_rate

        self._get_data(min_label, max_label)

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

    def _augment(self, augmenter: ia.meta) -> None:
        """
        :param augmenter:
        :return: None, appends augmented images to the train set.
        """
        train_aug = np.ndarray.astype(augmenter.augment_images(np.ndarray.astype(self.x_train * 256, np.uint8)), np.float32)
        train_aug = train_aug * (1 / 256)
        self.x_train = np.concatenate([self.x_train, train_aug])
        self.y_train = np.concatenate([self.y_train, self.y_train])
        for i in range(5):
            show_image(train_aug[i].reshape(NN_INPUT_SIZE))

    def _get_data(self, range_beg: int = 0, range_end: int = 52) -> None:
        """
        :param range_beg, range_end: only samples such that label \in [range_beg, range_end) will be
            used. Sensible values for (range_beg, range_end) would be:
            * 00, 52 -> to use eurecom only
            * 52, 78 -> to use ias_lab_rgbd_only
            * 78, 98 -> to use superface_dataset only
        :return: self.(x|y)_(train|test) are set as a result
        """

        # Load stored numpy arrays from files.
        logging.info("Loading data..")
        self.x_train = np.load(DB_LOCATION + '/gen/face_rotation_X_train.npy')
        self.y_train = np.load(DB_LOCATION + '/gen/face_rotation_Y_train.npy')
        self.x_test = np.load(DB_LOCATION + '/gen/face_rotation_X_test.npy')
        self.y_test = np.load(DB_LOCATION + '/gen/face_rotation_Y_test.npy')
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

        # Image augmentation.
        # aug1 = ia.CoarseSaltAndPepper(p=0.2, size_percent=0.05)
        # aug2 = ia.CoarseSaltAndPepper(p=0.4, size_percent=0.05)
        # aug3 = ia.PerspectiveTransform(scale=0.05)
        # aug4 = ia.PerspectiveTransform(scale=0.075)
        # aug5 = ia.Fliplr(1)
        # self._augment(aug1)
        # self._augment(aug2)
        # self._augment(aug3)
        # self._augment(aug4)
        # self._augment(aug5)

        logging.info("Loaded data..")

    def _visualize_kernels(self):
        """
        For each convolutional layer, visualizes filters and convolved images.
        """
        for layer_no in range(len(self.conv_layers)):
            num_filters = self.filters_count[layer_no]
            kernels = []
            applied_kernels = []
            for filter_no in range(num_filters):
                inp_x = NN_INPUT_SIZE[0] // (2 ** layer_no)
                inp_y = NN_INPUT_SIZE[1] // (2 ** layer_no)
                if layer_no == 0:
                    tmp_str = 'conv2d/kernel:0'
                else:
                    tmp_str = 'conv2d_%d/kernel:0' % layer_no
                kernel = [v for v in tf.global_variables() if v.name == tmp_str][0]
                kernel = kernel[:, :, :, filter_no]
                cur_conv_layer = self.conv_layers[layer_no]
                if layer_no == 0:
                    kernel = tf.reshape(kernel, [1] + self.kernel_size + [1])
                else:
                    kernel = tf.reshape(kernel, [1] +\
                                        [self.kernel_size[0], self.kernel_size[1] * self.filters_count[layer_no - 1]] +\
                                        [1])
                kernels.append(kernel)
                applied = tf.reshape(cur_conv_layer[0, :, :, filter_no], [1, inp_x, inp_y, 1])
                tf.summary.image('conv{0}_filter{1}_kernel'.format(layer_no, filter_no),
                                 kernel,
                                 family='kernels_layer{0}'.format(layer_no),
                                 max_outputs=1)
                tf.summary.image('conv{0}_filter{1}_applied'.format(layer_no, filter_no),
                                 applied,
                                 family='convolved_layer_{0}'.format(layer_no),
                                 max_outputs=1)
            concatenated_kernels = tf.concat(kernels, axis=2)
            # Write concatenated patches to summary.
            kernels_name = "kernels_layer{0}".format(layer_no)
            tf.summary.image(kernels_name,
                             concatenated_kernels,
                             family='kernels_all_layers')
        self.merged_summary = tf.summary.merge_all()

    def _visualize_exciting_patches(self):
        """
        For each convolutional layer, visualizes patches that excite each filter the most.
        """
        # Initialize fetch handles for exciting patches and their respective responses.
        self.exciting_patches = [[None] * k for k in self.filters_count]
        self.patches_responses = [[None] * k for k in self.filters_count]
        self.flattened_exciting_patches = [[None] * k for k in self.filters_count]
        self.all_exciting_patches_at_layer = [None for _ in self.filters_count]

        for layer_no in range(len(self.conv_layers)):
            num_filters = self.filters_count[layer_no]
            cur_conv_layer = self.conv_layers[layer_no]

            for filter_no in range(num_filters):
                # Find top 10 responses to current filter, in the current mini-batch.
                inp_x = NN_INPUT_SIZE[0] // (2 ** layer_no)
                inp_y = NN_INPUT_SIZE[1] // (2 ** layer_no)
                single_filtered_flattened = tf.reshape(cur_conv_layer[:, :, :, filter_no],
                                                       [self.mb_size * inp_x * inp_y])
                top10_vals, top10_indices = tf.nn.top_k(single_filtered_flattened,
                                                        k=10,
                                                        sorted=True)
                top10_reshaped = tf.map_fn(lambda sxy: [sxy // (inp_x * inp_y), (sxy // inp_y) % inp_x, sxy % inp_y],
                                           top10_indices,
                                           dtype=[tf.int32, tf.int32, tf.int32])

                def safe_cut_patch(sxy, size, img):
                    """
                    :param (sample_no, x, y)@sxy
                    :param size: size of patch to cut out
                    :param img: image to cut it from
                    :return: Cuts out a patch of size (|size|) located at (x, y) on
                        input #sample_no in current batch.
                    """
                    sample_no, x, y = sxy
                    pad_marg_x = (size[0] // 2) + 1 - (size[0] % 2)
                    pad_marg_y = (size[1] // 2) + 1 - (size[1] % 2)
                    padding = [[0, 0],
                               [pad_marg_x, pad_marg_x],
                               [pad_marg_y, pad_marg_y],
                               [0, 0]]
                    padded = tf.pad(img, padding)
                    return padded[sample_no, x:x + size[0], y:y + size[1], :]

                # Find patches corresponding to the top 10 responses.
                # Store patches and responses in class-visible array to be retrieved later.
                self.exciting_patches[layer_no][filter_no] = \
                    tf.map_fn(lambda sxy: safe_cut_patch(sxy,
                                                         size=(self.kernel_size[0] * (2 ** layer_no),
                                                               self.kernel_size[1] * (2 ** layer_no)),
                                                         img=self.x),
                              top10_reshaped,
                              dtype=tf.float32)
                self.patches_responses[layer_no][filter_no] = top10_vals

                # Flatten and concatenate the 10 patches to 2 dimensions for visualization.
                flattened_patches_shape = [1] + \
                                          [10 * self.kernel_size[0] * (2 ** layer_no),
                                           self.kernel_size[1] * (2 ** layer_no)] + \
                                          [1]
                # Write patches to summary.
                patch_name = "exciting_patches_filter{0}".format(filter_no)
                flattened_exciting_patches = tf.reshape(self.exciting_patches[layer_no][filter_no],
                                                        flattened_patches_shape,
                                                        name=patch_name)
                self.flattened_exciting_patches[layer_no][filter_no] = flattened_exciting_patches
            self.all_exciting_patches_at_layer[layer_no] = tf.concat(self.flattened_exciting_patches[layer_no], axis=2)
            # Write concatenated patches to summary.
            all_patches_name = "exciting_patches_layer{0}".format(layer_no)
            tf.summary.image(all_patches_name,
                             self.all_exciting_patches_at_layer[layer_no],
                             family='exciting_all_layers')

            # Merge all summaries.
            self.merged_summary = tf.summary.merge_all()

    def _create_model(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[self.mb_size] + NN_INPUT_SIZE + [1])
        self.y = tf.placeholder(dtype=tf.float32, shape=[self.mb_size, NUM_CLASSES])
        self.conv_layers = []

        signal = self.x

        for layer_no in range(len(self.filters_count)):
            num_filters = self.filters_count[layer_no]
            signal = tf.layers.batch_normalization(signal)

            # Init weights with std.dev = sqrt(2 / N)
            input_size = int(signal.get_shape()[1]) * int(signal.get_shape()[2]) * int(signal.get_shape()[3])
            w_init = tf.initializers.random_normal(stddev=sqrt(2 / input_size))

            # Convolutional layer
            cur_conv_layer = tf.layers.conv2d(inputs=signal,
                                              filters=num_filters,
                                              kernel_size=self.kernel_size,
                                              kernel_initializer=w_init,
                                              padding='same')

            # Reduce image dimensions in half.
            cur_pool_layer = tf.layers.max_pooling2d(inputs=cur_conv_layer,
                                                     pool_size=[2, 2],
                                                     strides=2)

            self.conv_layers.append(cur_conv_layer)

            # Set pooled image as current signal
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
                                              activation=tf.nn.leaky_relu,
                                              kernel_initializer=w_init)

            signal = cur_dense_layer

        # Apply last dense layer, with dropout
        cur_dropout_layer = tf.layers.dropout(inputs=signal,
                                              rate=self.dropout)

        # Init weights with std.dev = sqrt(2 / N)
        input_size = int(signal.get_shape()[1])
        w_init = tf.initializers.random_normal(stddev=sqrt(2 / input_size))
        signal = cur_dropout_layer
        cur_layer = tf.layers.dense(inputs=signal,
                                    activation=tf.nn.sigmoid,
                                    units=self.dense_layers[-1])
        signal = cur_layer

        self.preds = tf.argmax(signal, axis=1)
        self.loss = tf.losses.log_loss(self.y, signal)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.y, axis=1), tf.argmax(signal, axis=1)), tf.float32))
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        self.logger.info('list of variables {0}'.format(list(map(lambda x: x.name, tf.global_variables()))))

    def train_on_batch(self, batch_x, batch_y, global_step=1):
        """
        :return: [loss, accuracy]
        """

        # Write summaries every 100 steps
        if global_step % 100 == 0:
            results = self.sess.run([self.loss, self.accuracy, self.merged_summary, self.train_op],
                                    feed_dict={self.x: batch_x, self.y: batch_y})
            msum = results[2]
            self.writer.add_summary(msum, global_step=global_step)
            self.writer.flush()
        else:
            results = self.sess.run([self.loss, self.accuracy, self.train_op],
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
        # Update confusion matrix
        preds = results[2]
        for i in range(self.mb_size):
            self._confusion_matrix[np.argmax(batch_y[i]), preds[i]] += 1.

        return results[:2]

    def train_and_evaluate(self) -> None:
        """
        Train and evaluate model.
        """
        with tf.Session() as self.sess:
            # Initialize computation graph.
            self._create_model()
            # Add visualizations to computation graph.
            self._visualize_kernels()
            self._visualize_exciting_patches()

            # Initialize variables.
            tf.global_variables_initializer().run()

            # Initialize summary writer.
            self.writer = tf.summary.FileWriter(logdir='conv_vis')

            # Initialize progress bar.
            bar = Bar('', max=100, suffix='%(index)d/%(max)d ETA: %(eta)ds')

            for epoch_no in range(self.nb_epochs):
                # Train model on next batch

                batch = sample(list(range(self.x_train.shape[0])), self.mb_size)
                batch_x, batch_y = self.x_train[batch], self.y_train[batch]
                results = self.train_on_batch(batch_x, batch_y, global_step=epoch_no)

                bar.message = 'loss: {0[0]:.8f} acc: {0[1]:.3f} mean_acc: {1:.3f}'.\
                    format(results, np.mean(self.accs[-1000:]))

                if epoch_no % 100 == 0:
                    bar.finish()
                    bar = Bar('', max=100, suffix='%(index)d/%(max)d ETA: %(eta)ds')
                    batch_t = sample(list(range(self.x_test.shape[0])), self.mb_size)
                    batch_x_t, batch_y_t = self.x_test[batch_t], self.y_test[batch_t]
                    test_results = self.test_on_batch(batch_x_t, batch_y_t)
                    self.logger.info("(Test(batch):   Loss: {0[0]}, accuracy: {0[1]}, mean acc: {1}".
                                     format(test_results,
                                            np.mean(self.val_accs[-10:])))
                bar.next()

                if epoch_no % 1000 == 0:
                    # show_image(self._confusion_matrix)
                    pass


if __name__ == '__main__':
    # Test on eurecom only
    net = NeuralNet(mb_size=16,
                    filters_count=[10, 10],
                    min_label=0,
                    max_label=52)
    net.train_and_evaluate()

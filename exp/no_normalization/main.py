"""
Simple CNN for RGB+D face recognition.

"""

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys
from common.db_helper import DB_LOCATION

from common.tools import IMG_SIZE

if __name__ == '__main__':
    sys.setrecursionlimit(1000000)

    print("Loading data..")
    X_train = np.load('no_normalization_X_train.npy')
    Y_train = np.load('no_normalization_Y_train.npy')
    X_test = np.load('no_normalization_X_test.npy')
    Y_test = np.load('no_normalization_Y_test.npy')
    TOTAL_SUBJECTS_COUNT = Y_test[0].shape[0]
    print("Loaded data")
    # If you want, display the first input image. It is already scaled to [0;1]
    # tools.show_image(X_train[0].reshape((TYPES * IMG_SIZE, IMG_SIZE)));

    model = Sequential()
    model.add(Conv2D(20,
                     kernel_size=(6, 6),
                     activation='relu',
                     input_shape=(4 * IMG_SIZE, IMG_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(20,
                     kernel_size=(6, 6),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(20,
                     kernel_size=(6, 6),
                     activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.7))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(TOTAL_SUBJECTS_COUNT, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    train_generator = ImageDataGenerator().flow(X_train, Y_train)
    test_generator = ImageDataGenerator().flow(X_test, Y_test)

    total_epochs = 0
    while True:
        # 50 epochs is obviously not enough, repeat until convergence.
        model.fit_generator(train_generator,
                            initial_epoch=total_epochs,
                            epochs=total_epochs+50,
                            steps_per_epoch=10,
                            verbose=True,
                            validation_steps=1)
        loss = model.evaluate(X_test, Y_test)
        total_epochs += 50
        print("Epochs: %d;  Loss (on test set): %f;  Cat. accuracy: %f" % (total_epochs, loss[0], loss[1]))

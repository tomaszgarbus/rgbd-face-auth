"""
Simple CNN for RGB+D face recognition.

"""

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys
import logging
from common.db_helper import DB_LOCATION

from common.constants import IMG_SIZE

def get_model(model_id):
    # TODO sensible creating models
    if model_id == 0:
        logging.debug( "MODEL ZERO" )

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

        return model

    if model_id == 1:
        logging.debug( "MODEL ONE" )

        model = Sequential()

        #
        model.add(Conv2D(16, 16,
                     padding='same',
                     activation='relu',
                     input_shape=(4 * IMG_SIZE, IMG_SIZE, 1)))
        model.add(Conv2D(16, 16,
                     padding='same',
                     activation='relu'))
        model.add(MaxPooling2D(2))
        model.add(Dropout(1./16.))

        #
        model.add(Conv2D(32, 6,
                     padding='same',
                     activation='relu'))
        model.add(Conv2D(32, 6,
                     padding='same',
                     activation='relu'))
        model.add(MaxPooling2D(2))
        model.add(Dropout(0.25))

        #
        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.75))
        model.add(Dense(TOTAL_SUBJECTS_COUNT, activation='softmax'))

        return model

    raise ValueError("Incorrect model id")

def get_no_normalization_models():
    #TODO: move it to tools in smart way (argument with prefix name?)
    X_train = np.load(DB_LOCATION + '/gen/no_normalization_X_train.npy')
    Y_train = np.load(DB_LOCATION + '/gen/no_normalization_Y_train.npy')
    X_test = np.load(DB_LOCATION + '/gen/no_normalization_X_test.npy')
    Y_test = np.load(DB_LOCATION + '/gen/no_normalization_Y_test.npy')

    return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':
    sys.setrecursionlimit(1000000)

    logging.debug("Loading data..")
    X_train, Y_train, X_test, Y_test = get_no_normalization_models()

    TOTAL_SUBJECTS_COUNT = Y_test[0].shape[0]
    logging.debug("Loaded data")
    # If you want, display the first input image. It is already scaled to [0;1]
    # tools.show_image(X_train[0].reshape((TYPES * IMG_SIZE, IMG_SIZE)));

    model_id = 0
    if len(sys.argv) > 1:
        model_id = int(sys.argv[1])

    model = get_model(model_id)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    train_generator = ImageDataGenerator().flow(X_train, Y_train)
    test_generator = ImageDataGenerator().flow(X_test, Y_test) #TODO(Tomek): <--- ????

    total_epochs = 0
    epochs_per_round = 20

    while True:
        # 50 epochs is obviously not enough, repeat until convergence.
        # TODO: Infinity while isn't best option...
        model.fit_generator(train_generator,
                            initial_epoch=total_epochs,
                            epochs=total_epochs+epochs_per_round,
                            verbose=True,
                            validation_data=(X_test, Y_test)
        )
        loss = model.evaluate(X_test, Y_test)
        total_epochs += epochs_per_round
        logging.debug("Epochs: %d;  Loss (on test set): %f;  Cat. accuracy: %f" % (total_epochs, loss[0], loss[1]))

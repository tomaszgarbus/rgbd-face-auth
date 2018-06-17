from common.constants import DB_LOCATION
from classifiers.hog_classifier import HogFaceClassifier
from experiments.templates.load_database import load_data
from experiments.hogs_only.constants import EXP_NAME, INPUT_SIZE
from sklearn.metrics import accuracy_score
import numpy as np
import datetime

def from_hot_one(ys):
    return [np.argmax(y) for y in ys]


def run_main(params = None):
    if params is not None:
        print(str(datetime.datetime.now()))
        print("params = " + str(params))
    # test hogs only
    classifier = HogFaceClassifier(False, params)
    x_train, y_train, x_test, y_test = load_data(EXP_NAME, INPUT_SIZE)

    train_shape = x_train.shape
    test_shape = x_test.shape
    assert (test_shape[2] == 1 and test_shape[3] == 1)
    assert (train_shape[2] == 1 and train_shape[3] == 1)

    x_train = x_train.reshape(train_shape[0], train_shape[1])
    x_test = x_test.reshape(test_shape[0], test_shape[1])


#    score = classifier.test(x_train, from_hot_one(y_train))

    classifier.fit(x_train, from_hot_one(y_train))
    ys = classifier.prediction(x_test)
    evaluate = classifier.evaluate(x_test, y_test)
    score = evaluate.acc
    print("score is " + str(score))

    ys = classifier.prediction(x_train)
    score = accuracy_score(from_hot_one(y_train), ys)

    print("on train is " + str(score))

    np.save(DB_LOCATION + '/gen/' + EXP_NAME + '_pred_probs.npy', evaluate.pred_probs)

    return evaluate.pred_probs, y_test


if __name__ == '__main__':
    run_main()

from common.constants import DB_LOCATION
from classifiers.hog_classifier import HogFaceClassifier
from experiments.templates.load_database import load_data
from experiments.hogs_only.constants import EXP_NAME, INPUT_SIZE
from sklearn.metrics import accuracy_score
import numpy as np
import os

import experiments.no_rotation_channels_without_hogs.main as nn
import experiments.hogs_only.main as hogs


def test_ens(ws: tuple, out1: np.ndarray, out2: np.ndarray, y_test: np.ndarray, frames_limit=3) -> np.ndarray:
    (w1, w2) = ws
    wsum = w1 + w2
    w1 /= wsum
    w2 /= wsum

    out = np.add(np.multiply(out1, w1), np.multiply(out2, w2))
    voted_out = np.copy(out)
    for i in range(out.shape[0]):
        j = i
        while j < len(y_test) and np.equal(np.argmax(y_test[j]), np.argmax(y_test[i])) and j - i < frames_limit:
            j += 1
        voted_out[i] = np.mean(out[i:j], axis=0)
    out = voted_out
    out = np.apply_along_axis(lambda x: np.argmax(x), axis=1, arr=out)
    return out


def run_main(load_results=True):
    nn_file = DB_LOCATION + '/gen/' + nn.EXP_NAME + '_pred_probs.npy'
    hog_file = DB_LOCATION + '/gen/' + hogs.EXP_NAME + '_pred_probs.npy'
    if load_results and os.path.isfile(nn_file):
        print("Loading network results from file")
        nn_out = np.load(nn_file)
    else:
        nn_out = nn.run_main()
    if load_results and os.path.isfile(hog_file):
        print("Loading hog results from file")
        hog_out = np.load(hog_file)
        y_test = np.load(DB_LOCATION + '/gen/' + hogs.EXP_NAME + '_Y_test.npy')
    else:
        hog_out, y_test = hogs.run_main()
    nn_out = nn_out[:, :hog_out.shape[1]]
    print(hog_out.shape)

    test_probs = [
        (1, 2),
        (2, 3),
        (1, 1),
        (10, 9),
        (10, 8),
        (10, 7),
        (10, 6),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (8, 1),
        (9, 1),
        (10, 1),
        (100, 1),
        (1, 100)
    ]


    outs = list(map(lambda x: test_ens(x, nn_out, hog_out, y_test), test_probs))
    print(str(outs))

    for probs, out in zip(test_probs, outs):
        score = accuracy_score(hogs.from_hot_one(y_test), out)

        print("acc score is " + str(score) + " for voting weights: " + str(probs))

    # Visualize misclassified
    out = outs[0]
    labels = hogs.from_hot_one(y_test)
    x_test = np.load(DB_LOCATION + '/gen/' + nn.EXP_NAME + '_X_test.npy')
    misclassified = []
    for i in range(len(y_test)):
        if out[i] != labels[i]:
            misclassified.append(x_test[i][:, :, 0])
            print(labels[i])

    return outs, misclassified


if __name__ == '__main__':
    run_main()


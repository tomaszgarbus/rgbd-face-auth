from common.constants import NUM_CLASSES
from classifiers.hog_classifier import HogFaceClassifier
from experiments.templates.load_database import load_data
from experiments.hogs_only.constants import EXP_NAME, INPUT_SIZE
from sklearn.metrics import accuracy_score
import numpy as np

import experiments.no_rotation_channels_without_hogs.main as nn
import experiments.hogs_only.main as hogs


def test_ens(ws: tuple, out1: np.ndarray, out2: np.ndarray) -> np.ndarray:
    (w1, w2) = ws
    wsum = w1 + w2
    w1 /= wsum
    w2 /= wsum

    out = np.multiply(np.multiply(out1, w1), np.multiply(out2, w2))
    out = np.apply_along_axis(lambda x: np.argmax(x), axis=1, arr=out)
    return out


def run_main():
    # Use eurecom + ias_lab_rgbd, once we migrate to our own
    # dataset, NUM_CLASSES will be more meaningful
    nn_out = nn.run_main()
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
    ]

    outs = list(map(lambda x: test_ens(x, nn_out, hog_out), test_probs))
    print(str(outs))

    for probs, out in zip(test_probs, outs):
        score = accuracy_score(hogs.from_hot_one(y_test), out)

        print("acc score is " + str(score) + " for voting weights: " + str(probs))

    return outs


if __name__ == '__main__':
    run_main()


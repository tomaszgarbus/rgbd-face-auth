from sklearn.metrics import accuracy_score
import numpy as np

from common.constants import NUM_CLASSES
from classifiers.neural_net import NeuralNet
from classifiers.classification_results import ClassificationResults
import experiments.no_rotation_channels_without_hogs.constants as nn_exp_constants
import experiments.no_rotation_channels_without_hogs.main as nn
import experiments.hogs_only.main as hogs


def test_ens(ws: tuple, out1: np.ndarray, out2: np.ndarray) -> np.ndarray:
    (w1, w2) = ws
    wsum = w1 + w2
    w1 /= wsum
    w2 /= wsum

    out = np.add(np.multiply(out1, w1), np.multiply(out2, w2))
    out = np.apply_along_axis(lambda x: x, axis=1, arr=out)
    return out


def run_main():
    # Use eurecom + ias_lab_rgbd, once we migrate to our own
    # dataset, NUM_CLASSES will be more meaningful
    net = NeuralNet(experiment_name=nn_exp_constants.EXP_NAME,
                    input_shape=nn_exp_constants.NN_INPUT_SIZE,
                    mb_size=16,
                    kernel_size=[5, 5],
                    nb_epochs=50,
                    steps_per_epoch=1000,
                    filters_count=[10, 20, 20],
                    dense_layers=[1],
                    dropout_rate=0.5,
                    learning_rate=0.005,
                    binary_classification=True,
                    positive_class=0)
    nn_out = net.train_and_evaluate().pred_probs
    hog_out, y_test = hogs.run_main()
    hog_out = hog_out[:, 0]
    hog_out = hog_out.reshape([len(hog_out), 1])
    y_test = np.apply_along_axis(lambda x: float(x[0] == 1.), axis=1, arr=y_test)
    # nn_out = nn_out[:, :hog_out.shape[1]]
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

    outs = list(map(lambda x: test_ens(x, nn_out, hog_out), test_probs))
    print(str(outs))

    for probs, out in zip(test_probs, outs):
        print("Voting weights: " + str(probs))
        results = ClassificationResults(pred_probs=out, labels=y_test, binary=True)
        for prec in [0.9, 0.99, 0.995, 0.999, 1]:
            print("Recall for precision " + str(prec) + ": " + str(results.get_recall_for_precision(prec)))

    return outs


if __name__ == '__main__':
    run_main()


from sklearn.metrics import accuracy_score
import numpy as np
import os

from common.constants import DB_LOCATION
from classifiers.neural_net import NeuralNet
from classifiers.classification_results import ClassificationResults
import experiments.no_rotation_channels_without_hogs.constants as nn_exp_constants
import experiments.no_rotation_channels_without_hogs.main as nn
import experiments.hogs_only.main as hogs


def test_ens(ws: tuple, out1: np.ndarray, out2: np.ndarray, y_test:np.ndarray, frames_limit=3) -> np.ndarray:
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
    out = np.apply_along_axis(lambda x: x, axis=1, arr=out)
    return out


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


def run_main(pos_class=0, net_iters=20):
    nn_file = DB_LOCATION + '/gen/' + nn.EXP_NAME + '_binary_pred_probs_' + str(pos_class) + '.npy'
    nn_ckpt = 'ckpts/' + nn.EXP_NAME + '_' + str(pos_class) + '.npy'
    hog_file = DB_LOCATION + '/gen/' + hogs.EXP_NAME + '_pred_probs.npy'
    if os.path.isfile(nn_file):
        print("Loading network results from file")
        nn_out = np.load(nn_file)
    else:
        net = NeuralNet(experiment_name=nn_exp_constants.EXP_NAME,
                        input_shape=nn_exp_constants.NN_INPUT_SIZE,
                        mb_size=16,
                        kernel_size=[5, 5],
                        nb_epochs=net_iters,
                        steps_per_epoch=1000,
                        filters_count=[20, 20, 40],
                        dense_layers=[1],
                        dropout_rate=0.5,
                        learning_rate=0.005,
                        binary_classification=True,
                        positive_class=pos_class,
                        ckpt_file=nn_ckpt)
        nn_out = net.train_and_evaluate().pred_probs
        np.save(nn_file, nn_out)
    if os.path.isfile(hog_file):
        print("Loading hog results from file")
        hog_out = np.load(hog_file)
        y_test = np.load(DB_LOCATION + '/gen/' + hogs.EXP_NAME + '_Y_test.npy')
    else:
        hog_out, y_test = hogs.run_main()
    hog_out = hog_out[:, pos_class]
    hog_out = hog_out.reshape([len(hog_out), 1])
    y_test = np.apply_along_axis(lambda x: float(x[pos_class] == 1.), axis=1, arr=y_test)
    # nn_out = nn_out[:, :hog_out.shape[1]]

    outs = list(map(lambda x: test_ens(x, nn_out, hog_out, y_test), test_probs))

    for probs, out in zip(test_probs, outs):
        print("Voting weights: " + str(probs))
        results = ClassificationResults(pred_probs=out, labels=y_test, binary=True)
        for prec in [0.9, 0.99, 0.995, 0.999, 1]:
            print("Recall for precision " + str(prec) + ": " + str(results.get_recall_for_precision(prec)))

    return outs, y_test


if __name__ == '__main__':
    all_pred_probs = [[] for i in range(len(test_probs))]
    all_y_test = []
    for i in range(5):
        print("running with positive class " + str(i))
        outs, y_test = run_main(pos_class=i, net_iters=20)
        all_y_test += list(y_test)
        for j in range(len(test_probs)):
            all_pred_probs[j] += list(outs[j])

    for pred_probs, vote_props in zip(all_pred_probs, test_probs):
        results = ClassificationResults(pred_probs=pred_probs, labels=all_y_test, binary=True)
        for prec in [0.9, 0.99, 0.995, 0.999, 1]:
            print("Recall for precision " + str(prec) + ": " + str(results.get_recall_for_precision(prec)))


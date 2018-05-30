from common.constants import NUM_CLASSES
from classifiers.neural_net import NeuralNet
from experiments.no_rotation_channels_with_hogs.constants import EXP_NAME, NN_INPUT_SIZE

if __name__ == '__main__':
    # Test on eurecom + ias_lab_rgbd
    net = NeuralNet(experiment_name=EXP_NAME,
                    input_shape=NN_INPUT_SIZE,
                    mb_size=16,
                    kernel_size=[5, 5],
                    nb_epochs=50,
                    steps_per_epoch=1000,
                    filters_count=[10, 20, 20],
                    dense_layers=[NUM_CLASSES],
                    dropout_rate=0.5,
                    learning_rate=0.05,
                    min_label=0,
                    max_label=78)
    net.train_and_evaluate()
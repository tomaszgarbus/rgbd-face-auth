# Classifiers
## Classification Results
Unified class for classficiation results, that both classifier returns.
It stores predictins, raw predictions (probabilities), accuracy, mode (binary or multi-class). 

## Convolutional Neural Net
Works in two modes (binary and multi-class). Highly customizable, for options
refer to the constructor comments. An important parameter is `experiment_name`,
because the `NeuralNet` class handles loading input data from `/face_auth/database/gen/{experiment_name}_*`.
Implemented in TensorFlow.

## HOGs Classifier
Implemented in scikit-learn classifier combining SVMs and ExtraTrees. The parameters
are already adjusted to what gave the best results in conducted experiments.
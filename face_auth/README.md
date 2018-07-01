This directory contains experiments related to face detection, normalization and recognition. 

# Database
Most of the code in here assumes that you have a symbolic link under `/face_auth/database`, which
links to the database.

Should you want to create any processed files in your experiments, please write them to
`/face_auth/database/gen` and append the name of your experiment as a prefix (e.g.
`/face_auth/database/gen/my_experiment_X_train.npy`).

# Classifiers
Implementation of used classifiers: NeuralNet, HogFaceClassifier and unified
class for classification results - ClassificationResults.

# Common
Tools, mainly for image manipulation, constants.

# Controller
Controller-layer code for Face class.

# Experiments
All experiments are here. You can create your own as long as they don't affect other directories in any way.

# Face-rotation
All code relating facial rotations - trimming, normalization, dropping corner values,
detecting current angle.

# Model
Model-layer code for Face class.
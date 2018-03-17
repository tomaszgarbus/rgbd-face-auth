This directory contains experiments related to face detection, normalization and recognition. 

# Database
Most of the code in here assumes that you have a symbolic link under `/exp/database`, which
links to the unzipped latest (`face_rgbd_data_light_v3.zip`) version of our database.

# Common
Directory `common` contains code used by all experiments. You can write new functions to `tools.py`
or `db_helper.py`, if they may be useful in more than one experiment. Be mindful not to change the
code already used by some experiments.

## DBHelper
`DBHelper` and `Database` classes in `db_helper.py` file are utilities for not only scanning and
loading the dataset, but it also has a helper function to cut out the faces from the images, using
external library.

# Experiment 1: No normalization
This experiment was inspired by the paper http://www.iab-rubric.org/papers/PID2857163.pdf
Tested approach was to cut out faces from the images, using external library, add entropy maps and
run simple CNNs. Despite using smaller dataset, less preprocessing (no saliency maps or HOGs)
and very little experiments with the network model itself, we achieved very similar top1 accuracy
as the paper - around 0.89 categorical accuracy.

Thus, we found that running filters on the input images is not enough. Moving on to a smarter
normalization.

# Experiment 2: Face rotation
**TODO** In this approach, we want to find landmarks on the face and use them to calculate the
angle of the face. Then, we will rotate the face to a frontal position. Thanks to depth images,
we can represent the face as a cloud of points (X, Y, Z, R, G, B) and rotate with simple transformation
once the angle is found. 
This directory contains experiments related to face detection, normalization and recognition. 

# Database
Most of the code in here assumes that you have:
* a symbolic link under `/exp/database`, which
links to the unzipped latest (`face_rgbd_data_light_v3.zip`) version of our database.
* Eurecom dataset extracted to `/exp/database` 

Should you want to create any processed files in your experiments, please write them to
`/exp/database/gen` and append the name of your experiment as a prefix (e.g.
`/exp/database/gen/my_experiment_X_train.npy`).

# Inside `/exp/experiments`
All experiments are here. You can create your own as long as they don't affect other directories in any way.
## Creating your experiment
1. Choose a name for your experiment, from now on we refer to it as "`{my_name}`"
2. Create a directory named "`/exp/experiments/{my_exp}`"
2. Do whatever you want *inside* this directory. If you wish to generate any larger temporary files, they
must match the pattern "`/exp/database/gen/{my_exp}_*`".

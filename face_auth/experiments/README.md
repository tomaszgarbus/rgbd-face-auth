# Experiments
THIS IS A PLAYGROUND. The code in this directory may be not formatted, not type-hinted
or even not working as expected. We have decided to leave this directory to show
how we ran our final experiments and how to create new ones.

All experiments are here. You can create your own as long as they don't affect other directories in any way.
## Creating your experiment
1. Choose a name for your experiment, from now on we refer to it as "`{my_exp}`"
2. Create a directory named "`/face_auth/experiments/{my_exp}`"
2. Do whatever you want *inside* this directory. If you wish to generate any larger temporary files, they
must match the pattern "`/face_auth/database/gen/{my_exp}_*`".

## Preprocess-Run pattern
All our experiments follow preprocess-run design.
1. Preprocess: Each experiment contains `preprocess_input.py` file, which uses
`InputPreprocessor` class from `/face_auth/experiments/templates/base_preprocess_input.py`. As a result,
preprocessed input is cached as numpy arrays in `/face_auth/database/gen/{my_exp}_*`.
2. Run: The second step of running an experiment is executing its `main.py` file, which
loads the preprocessed input and feeds it into a classifier.

Note that you can create an experiment that does not follow this pattern.

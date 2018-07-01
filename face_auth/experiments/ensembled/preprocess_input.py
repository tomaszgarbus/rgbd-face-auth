"""
    only hogs
"""
import experiments.hogs_only.preprocess_input as hogs
import experiments.no_rotation_channels_without_hogs.preprocess_input as nn


def run_preprocess():
    hogs.run_preprocess()
    nn.run_preprocess()


if __name__ == '__main__':
    run_preprocess()

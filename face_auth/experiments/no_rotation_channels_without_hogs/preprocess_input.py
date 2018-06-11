"""
    4 channels: grey_or_ir, depth, grey_or_ir_entropy, depth_entropy
"""
import numpy as np

from experiments.templates.base_preprocess_input import InputPreprocessor
from experiments.no_rotation_channels_without_hogs.constants import EXP_NAME, NN_INPUT_SIZE
from controller.normalization import normalized, hog_and_entropy


def build_input_vector(face):
    (gir_face, depth_face) = (face.gir_img, face.depth_img)
    if gir_face is None or depth_face is None:
        return None
    if np.isnan(gir_face).any() or np.isnan(depth_face).any():
        return None
    try:
        face = normalized(face, rotate=False)
        face = hog_and_entropy(face)
    except ValueError:
        return None
    return face.get_channels()[:, :, :4]


def run_preprocess():
    preprocessor = InputPreprocessor(exp_name=EXP_NAME,
                                     nn_input_size=NN_INPUT_SIZE,
                                     build_input_vector=build_input_vector)
    preprocessor.preprocess()


if __name__ == '__main__':
    run_preprocess()

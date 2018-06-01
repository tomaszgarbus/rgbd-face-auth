"""
    only hogs
"""
import numpy as np

from experiments.templates.base_preprocess_input import InputPreprocessor
from experiments.hogs_only.constants import EXP_NAME, INPUT_SIZE
from controller.normalization import normalized, hog_and_entropy


def build_input_vector(face):
    (grey_face, depth_face) = (face.grey_img, face.depth_img)
    if grey_face is None or depth_face is None:
        return None
    if np.isnan(grey_face).any() or np.isnan(depth_face).any():
        return None
    try:
        face = normalized(face, rotate=False)
        face = hog_and_entropy(face)
    except ValueError:
        return None
    return face.get_fd_desc()


if __name__ == '__main__':
    preprocessor = InputPreprocessor(exp_name=EXP_NAME,
                                     nn_input_size=INPUT_SIZE,
                                     build_input_vector=build_input_vector)
    preprocessor.preprocess()

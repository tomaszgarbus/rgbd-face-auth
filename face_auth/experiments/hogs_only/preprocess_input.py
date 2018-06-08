"""
    only hogs
"""
import numpy as np
from imgaug import augmenters as ia

from experiments.templates.base_preprocess_input import InputPreprocessor
from experiments.hogs_only.constants import EXP_NAME, INPUT_SIZE
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
    return face.get_fd_desc()


augmenters = [
    ia.Noop(),
    ia.CoarseSaltAndPepper(p=0.2, size_percent=0.30),
    ia.CoarseSaltAndPepper(p=0.4, size_percent=0.30),
    ia.Pad(px=(3, 0, 0, 0)),
    ia.Pad(px=(0, 3, 0, 0)),
    ia.Pad(px=(0, 0, 3, 0)),
    ia.Pad(px=(0, 0, 0, 3)),
    ia.GaussianBlur(sigma=0.25),
    ia.GaussianBlur(sigma=0.5),
    ia.GaussianBlur(sigma=1),
    ia.GaussianBlur(sigma=2),
    ia.Affine(rotate=-2),
    ia.Affine(rotate=2)
]

if __name__ == '__main__':
    preprocessor = InputPreprocessor(exp_name=EXP_NAME,
                                     nn_input_size=INPUT_SIZE,
                                     build_input_vector=build_input_vector)
    preprocessor.preprocess()

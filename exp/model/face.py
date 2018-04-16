import numpy as np
import face_recognition

from common import tools
from common.constants import IMG_SIZE


class Face:
    grey_img: np.ndarray((IMG_SIZE, IMG_SIZE)) = None
    depth_img: np.ndarray((IMG_SIZE, IMG_SIZE)) = None
    """ maps pixels which belong to the face """
    mask: np.ndarray((IMG_SIZE, IMG_SIZE), dtype=np.bool) = None
    landmarks = None

    """ list of points defining the face surface"""
    face_points: dict = None
    """ center of face """
    face_center: tuple = None
    """ vector orthogonal to the face surface"""
    azimuth: tuple = None

    preprocessed: tuple = False

    def __init__(self, grey_img: np.array, depth_img: np.array):
        self.grey_img = grey_img
        self.depth_img = depth_img
        if self.grey_img is not None and self.depth_img is not None:
            assert self.depth_img.shape == self.grey_img.shape == (IMG_SIZE, IMG_SIZE)
        self._preprocess_landmarks()

    _iter_rq = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter_rq == 0:
            self._iter_rq += 1
            return self.grey_img
        elif self._iter_rq == 1:
            self._iter_rq += 1
            return self.depth_img
        else:
            self._iter_rq = 0
            raise StopIteration

    def _preprocess_landmarks(self) -> None:
        tmp = face_recognition.face_landmarks((self.grey_img * 256).astype(np.uint8))
        if not tmp:
            return
        assert (len(tmp) == 1)  # allowing to recognise only one face
        tmp = tmp[0]
        # swapping coordinates to suitable format
        self.landmarks = {k: list([(x, y) for (y, x) in v]) for k, v in tmp.items()}

    def show_grey(self) -> None:
        tools.show_image(self.grey_img)

    def show_depth(self) -> None:
        tools.show_image(self.depth_img)

    """ shows face with face points and azimuth"""
    def show_position(self) -> None:
        tools.show_position(self.grey_img, self.face_points, self.azimuth, self.face_center)


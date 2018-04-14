import numpy as np
import face_recognition

from common import tools

class Face:
    grey_img = None
    depth_img = None
    landmarks = None

    def __init__(self, grey_img: np.array, depth_img: np.array):
        self.grey_img = grey_img
        self.depth_img = depth_img

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

    def _find_landmarks(self):
        tmp = face_recognition.face_landmarks(self.grey_img)
        if not tmp:
            return
        self.landmarks = tmp

    def show_grey(self) -> None:
        tools.show_image(self.grey_img)

    def show_depth(self) -> None:
        tools.show_image(self.depth_img)
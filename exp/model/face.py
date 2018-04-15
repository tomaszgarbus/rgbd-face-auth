import numpy as np
import face_recognition

from common.constants import IMG_SIZE
from common import tools


class Face:
    grey_img: np.ndarray((IMG_SIZE, IMG_SIZE)) = None
    depth_img: np.ndarray((IMG_SIZE, IMG_SIZE)) = None
    """ maps pixels which belong to the face """
    mask: np.ndarray((IMG_SIZE, IMG_SIZE), dtype=np.bool) = None
    landmarks = None

    """ list of points defining the face surface"""
    face_points = None
    """ center of face """
    face_center = None
    """ vector orthogonal to the face surface"""
    azimuth = None

    def __init__(self, grey_img: np.array, depth_img: np.array):
        self.grey_img = grey_img
        self.depth_img = depth_img
        if self.grey_img is not None and self.depth_img is not None:
            assert self.depth_img.shape == self.grey_img.shape == (IMG_SIZE, IMG_SIZE)

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

    def _find_landmarks(self) -> None:
        tmp = face_recognition.face_landmarks(self.grey_img)
        if not tmp:
            return
        assert (len(tmp) == 1)  # allowing to recognise only one face
        tmp = tmp[0]
        #  swapping coordinates to suitable format
        self.landmarks = {k: list([(x, y) for (y, x) in v]) for k, v in tmp.iter()}

    def show_grey(self) -> None:
        tools.show_image(self.grey_img)

    def show_depth(self) -> None:
        tools.show_image(self.depth_img)

    """ shows face with face points and azimuth"""
    def show_position(self) -> None:
        tools.show_position(self.grey_img, self.face_points, self.azimuth, self.face_center)

    def normalization(self):
        # Trim face
        trim_face.trim_greyd(face)
        img_grey, img_depth = face

        # Display trimmed photo
        face.show_grey()
        face.show_depth()

        rotate.drop_corner_values(face.depth_img, face.grey_img)

        # Display the photo after normalizing mean
        face.show_grey()
        face.show_depth()

        # Find the angle
        rotation, face_points = find_angle(face)
        if rotation is None:
            continue
        center = face_points["forehead"]
        print("center = " + str(center))

        # Apply rotation
        rotated_face, face_points = rotate.rotate_greyd_img(face, rotation, face_points)


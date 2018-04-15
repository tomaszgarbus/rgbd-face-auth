import numpy as np
import face_recognition

from common import tools
from common.constants import IMG_SIZE
from face_rotation.trim_face import trim_greyd
from face_rotation.rotate import rotate_greyd_img
from face_rotation.other import construct_face_points, drop_corner_values
from face_rotation.find_angle import find_angle
from face_rotation.recentre import recentre


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

    def preprocessing(self) -> None:
        if self.preprocessed:
            return
        self.preprocessed = True

        # Display the original photo
        self.show_grey()
        self.show_depth()

        # Trim face
        trim_greyd(self)

        # Display trimmed photo
        self.show_grey()
        self.show_depth()

        # Drop corner values and rescale to 0...1
        drop_corner_values(self)

        # Calculate face center and points defining face surface
        construct_face_points(self)

        # Display the photo after normalizing mean
        self.show_position()
        self.show_depth()

    def normalized(self) -> 'Face':
        self.preprocessing()

        # Find the angle
        rotation = find_angle(self)
        if rotation is None:
            return None

        # Apply rotation
        rotated_face = rotate_greyd_img(self, rotation)

        # Display the results
        rotated_face.show_position()
        rotated_face.show_depth()

        # centering
        rotated_face.show_position()
        recentre(rotated_face)
        rotated_face.show_position()

        # tools.show_3d_plot(rotate.to_one_matrix(rotated_face))



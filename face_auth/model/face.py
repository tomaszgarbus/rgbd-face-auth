import numpy as np
import face_recognition
from typing import Optional

from common import tools
from common.constants import IMG_SIZE


class Face:
    """ grey or ir image """
    gir_img: np.ndarray((IMG_SIZE, IMG_SIZE)) = None
    depth_img: np.ndarray((IMG_SIZE, IMG_SIZE)) = None
    rgb_img: np.ndarray((IMG_SIZE, IMG_SIZE)) = None
    """ maps pixels which belong to the face """
    mask: np.ndarray((IMG_SIZE, IMG_SIZE), dtype=np.bool) = None
    landmarks = None

    entropy_map_gir_image: np.ndarray((IMG_SIZE, IMG_SIZE)) = None
    entropy_map_depth_image: np.ndarray((IMG_SIZE, IMG_SIZE)) = None
    hog_depth_image: np.ndarray((IMG_SIZE, IMG_SIZE)) = None
    hog_gir_image: np.ndarray((IMG_SIZE, IMG_SIZE)) = None
    hog_depth_fd: np.ndarray = None
    hog_gir_fd: np.ndarray = None

    """ list of points defining the face surface"""
    face_points: dict = None
    """ center of face """
    face_center: tuple = None
    """ vector orthogonal to the face surface"""
    azimuth: tuple = None

    preprocessed: tuple = False

    def __init__(self, gir_img: np.array, depth_img: np.array, rgb_img: Optional[np.array] = None):
        self.gir_img = gir_img
        self.depth_img = depth_img
        self.rgb_img = rgb_img
        if self.gir_img is not None and self.depth_img is not None:
            assert self.depth_img.shape == self.gir_img.shape == (IMG_SIZE, IMG_SIZE)
            self._preprocess_landmarks()

    _iter_rq = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter_rq == 0:
            self._iter_rq += 1
            return self.gir_img
        elif self._iter_rq == 1:
            self._iter_rq += 1
            return self.depth_img
        else:
            self._iter_rq = 0
            raise StopIteration

    def _preprocess_landmarks(self) -> None:
        tmp = face_recognition.face_landmarks((self.gir_img * 256).astype(np.uint8))
        if not tmp:
            return
        assert (len(tmp) == 1)  # allowing to recognise only one face
        tmp = tmp[0]
        # swapping coordinates to suitable format
        self.landmarks = {k: list([(x, y) for (y, x) in v]) for k, v in tmp.items()}

    def get_concat(self) -> np.ndarray:
        return tools.concat_images([self.gir_img, self.depth_img,
                                    self.entropy_map_gir_image, self.entropy_map_depth_image,
                                    self.hog_gir_image, self.hog_depth_image])

    def get_fd_desc(self):
        return np.concatenate((self.hog_gir_fd, self.hog_depth_fd), axis=0)

    def get_channels(self) -> np.ndarray:
        channels = np.zeros((IMG_SIZE, IMG_SIZE, 6))
        channels[:, :, 0] = self.gir_img
        channels[:, :, 1] = self.depth_img
        channels[:, :, 2] = self.entropy_map_gir_image
        channels[:, :, 3] = self.entropy_map_depth_image
        channels[:, :, 4] = self.hog_gir_image
        channels[:, :, 5] = self.hog_depth_image
        return channels

    def show_grey_or_ir(self) -> None:
        tools.show_image(self.gir_img)

    def show_depth(self) -> None:
        tools.show_image(self.depth_img)

    """ shows face with face points and azimuth"""
    def show_position(self) -> None:
        tools.show_position(self.gir_img, self.face_points, self.azimuth, self.face_center)

    def show_map_and_entropy(self):
        tools.show_image(self.entropy_map_gir_image)
        tools.show_image(self.hog_gir_image)
        tools.show_image(self.entropy_map_depth_image)
        tools.show_image(self.hog_depth_image)


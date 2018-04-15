import numpy as np
import math
from common.constants import IMG_SIZE, BGCOLOR, SMOOTHEN_ITER
import random
import logging

from model.face import Face


def _rx(theta: float) -> np.ndarray:
    """ returns rotation matrix for x axis """
    return np.array([[1, 0, 0],
                     [0, math.cos(theta), -math.sin(theta)],
                     [0, math.sin(theta), math.cos(theta)]])


def _ry(theta: float) -> np.ndarray:
    """ returns rotation matrix for y axis """
    return np.array([[math.cos(theta), 0, math.sin(theta)],
                     [0, 1, 0],
                     [-math.sin(theta), 0, math.cos(theta)]])


def _rz(theta: float) -> np.ndarray:
    """ returns rotation matrix for z axis """
    return np.array([[math.cos(theta), -math.sin(theta), 0],
                     [math.sin(theta), math.cos(theta), 0],
                     [0, 0, 1]])


def _rescale_one_dim(_points: np.ndarray) -> None:
    """
        Scales one dimension into interval 0..1
    """
    _points -= _points.min()
    if _points.max() > 0:
        _points /= _points.max()


def _rescale(_points: np.ndarray) -> None:
    """
        Scales each dimension into interval 0..1
    """
    for i in range(_points.shape[-1]):
        _rescale_one_dim(_points[:, :, i])


def normalize_face_points(_points: np.ndarray, face_points: dict, rotation_matrix: np.ndarray((3, 3))) -> dict:
    """
        Scales face points into interval 0..1
    """
    face_points = {k: np.dot(rotation_matrix, np.asarray(v)) for k, v in face_points.items()}
    face_points = {k: (v.item(0), v.item(1), v.item(2)) for k, v in face_points.items()}
    for i in range(_points.shape[-1] - 1):
        min = _points[:, :, i].min()
        for k, v in face_points.items():
            v = list(v)
            v[i] -= min
            face_points[k] = tuple(v)
        max = _points[:, :, i].max()
        if _points.max() > 0:
            for k, v in face_points.items():
                v = list(v)
                v[i] /= max
                face_points[k] = tuple(v)
    return face_points


def _median_neighbors(points: np.ndarray,
                      center: tuple((int, int)),
                      size: int,
                      min_value: float,
                      max_value: float) -> float:
    """
        Find a median of those points which have depth in range
        |min_value|..|max_value|, only in the neighborhood of |center|
        of size |size|.
        :param X: array of points, of size (IMG_SIZE, IMG_SIZE)
        :param center: center - a tuple (x, y)
        :param size: maximum distance from the center
        :param min_value: minimum value to be considered
        :param max_value: maximum value to be considered
        :return: a single float value
    """
    min_x = max(0, center[0]-size)
    max_x = min(IMG_SIZE, center[0] + size)
    min_y = max(0, center[1] - size)
    max_y = min(IMG_SIZE, center[1] + size)
    possible_ok_points = (max_x - min_x)*(max_y - min_y)

    vals = points[min_x:max_x,min_y:max_y]
    vals = vals[vals >= min_value]
    vals = vals[vals <= max_value]
    if vals.size == 0 or vals.size <= (1/2)*possible_ok_points:
        # Giving up, returning 0 (to be handled in next iter)
        return 0
    return np.median(vals)


def _smoothen(img: np.ndarray) -> np.ndarray:
    """ smoothens the rotated image, i.e. fills each empty pixel with a median
    of non-empty neighboring pixels """
    order_i = list(range(IMG_SIZE))
    order_j = list(range(IMG_SIZE))
    random.shuffle(order_i)
    random.shuffle(order_j)
    for i in order_i:
        for j in order_j:
            if img[i, j] == 0:
                img[i, j] = _median_neighbors(img, (i, j), 1, 0.01, 0.9)
    return img


def drop_corner_values(dimage: np.ndarray,
                       image: np.ndarray,
                       lower_threshold : float = 0.1,
                       upper_threshold : float = 0.98) -> None:
    """
        Erase those pixels which are too close or to far to be treated as
        valuable data.
    """
    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            # Replace pixels beyond reasonable value range with median of closest
            # "good" pixels
            if dimage[i, j] > upper_threshold or dimage[i, j] < lower_threshold:
                new_value = _median_neighbors(dimage[:, :], (i, j), IMG_SIZE // 3, lower_threshold,
                                                    upper_threshold)
                if new_value > upper_threshold or new_value < lower_threshold:
                    new_value = 0.5
                dimage[i, j] = new_value

    # Scale each dimension into interval 0..1
    _rescale_one_dim(dimage)
    _rescale_one_dim(image)


def to_one_matrix(face: Face) -> np.ndarray:
    points = np.zeros((IMG_SIZE, IMG_SIZE, 4))
    for i in range(IMG_SIZE):
        points[i, :, 0] = i
        points[:, i, 1] = i
        points[i, :, 2] = face.depth_img[i, :]
        points[i, :, 3] = face.grey_img[i, :]
    return points


def rotate_greyd_img(face: Face, rotation_matrix: np.ndarray, face_points):
    # First, we prepare the matrix X of points (x, y, z, Grey)
    (grey_img, depth_img) = (face)
    points = to_one_matrix(face)

    drop_corner_values(points[:, :, 2], points[:, :, 3])

    # Normalize x an y dimensions of |points|
    _rescale_one_dim(points[:, :, 0])
    _rescale_one_dim(points[:, :, 1])

    # Rotate around each axis
    # rotation_matrix = np.matmul(_rx(theta_x), np.matmul(_ry(theta_y), _rz(theta_z)))
    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            points[i, j, :3] = np.dot(rotation_matrix, points[i, j, :3].reshape(3, 1)).reshape(3)

    # Normalize once more after rotation
    face_points = normalize_face_points(points, face_points, rotation_matrix)
    _rescale(points)

    # Apply rotated image to grey and depth photo
    grey_rotated = np.zeros((IMG_SIZE, IMG_SIZE))
    depth_rotated = np.zeros((IMG_SIZE, IMG_SIZE))
    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            if np.isnan(points[i, j, 0]) or np.isnan(points[i, j, 1]):
                logging.warning("Unexpected NaN in rotated image -- skipping invalid pixel")
                continue
            x = int(points[i, j, 0] * (IMG_SIZE - 1))
            y = int(points[i, j, 1] * (IMG_SIZE - 1))
            if x < 0 or y < 0 or x >= IMG_SIZE or y >= IMG_SIZE:
                continue
            g = points[i, j, 3]
            z = points[i, j, 2]
            if depth_rotated[x, y] < z:
                grey_rotated[x, y] = g
                depth_rotated[x, y] = z

    for i in range(SMOOTHEN_ITER):
        grey_rotated = _smoothen(grey_rotated)
        depth_rotated = _smoothen(depth_rotated)

    # If you want to view the rotated image, use the following:
    # tools.show_image(grey_rotated)
    # tools.show_image(depth_rotated)
    # Or:
    # tools.show_3d_plot(points)
    return (Face(grey_rotated, depth_rotated), face_points)


def rotate_greyd_img_by_angle(face: Face,
                              face_points,
                              theta_x : float = 0,
                              theta_y : float = 0,
                              theta_z : float = 0,):
    rotation_matrix = np.matmul(_rx(theta_x), np.matmul(_ry(theta_y), _rz(theta_z)))
    return rotate_greyd_img(face, rotation_matrix, face_points)

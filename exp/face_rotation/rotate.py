import numpy as np
import math
from common.tools import IMG_SIZE


def _rx(theta):
    """ returns rotation matrix for x axis """
    return np.array([[1, 0, 0],
                     [0, math.cos(theta), -math.sin(theta)],
                     [0, math.sin(theta), math.cos(theta)]])


def _ry(theta):
    """ returns rotation matrix for y axis """
    return np.array([[math.cos(theta), 0, math.sin(theta)],
                     [0, 1, 0],
                     [-math.sin(theta), 0, math.cos(theta)]])


def _rz(theta):
    """ returns rotation matrix for z axis """
    return np.array([[math.cos(theta), -math.sin(theta), 0],
                     [math.sin(theta), math.cos(theta), 0],
                     [0, 0, 1]])


def _smoothen(img):
    """ smoothens the rotated image, i.e. fills each empty pixel with a median
    of non-empty neighboring pixels """
    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            if img[i, j] == 0:
                img[i, j] = 0.7
                # TODO(ludziej): apply median from non-empty neighbors
    return img


def _normalize(_points):
    """ scales each dimension into interval 0..1 """
    for i in range(_points.shape[-1]):
        _points[:, :, i] -= _points[:, :, i].min()
        if _points[:, :, i].max() > 0:
            _points[:, :, i] /= _points[:, :, i].max()


def _median_depth(points, center, size, min_value, max_value):
    """
    Find a median of those points which have depth in range
    |min_value|..|max_value|, only in the neighborhood of |center|
    of size |size|.
    :param X: array of points, of size (IMG_SIZE, IMG_SIZE, 3)
    :param center: center - a tuple (x, y)
    :param size: maximum distance from the center
    :param min_value: minimum value of depth to be considered
    :param max_value: maximum value of depth to be considered
    :return: a single float value
    """
    min_x = max(0, center[0]-size)
    max_x = min(IMG_SIZE, center[0] + size)
    min_y = max(0, center[1] - size)
    max_y = min(IMG_SIZE, center[1] + size)
    vals = points[min_x:max_x,min_y:max_y,2]
    vals = vals[vals >= min_value]
    vals = vals[vals <= max_value]
    if vals.size == 0:
        # Giving up, returning median of entire cloud of points
        return np.median(points[:, :, 2])
    return np.median(vals)

def rotate_greyd_img(greyd_img, theta_x=0, theta_y=0, theta_z=0):
    """
    :param greyd_img: a tuple (grey_image, depth_image).
        images are required to be of size tools.IMG_SIZE
    :param theta_x: angle of rotation around axis x
    :param theta_y: angle of rotation around axis y
    :param theta_z: angle of rotation around axis z
    :return: tuple (grey_image, depth_image), rotated by requested angles
    """
    # First, we prepare the matrix X of points (x, y, z, Grey)
    (grey_img, depth_img) = (greyd_img)
    points = np.zeros((IMG_SIZE, IMG_SIZE, 4))
    for i in range(IMG_SIZE):
        points[i, :, 0] = i
        points[:, i, 1] = i
        points[i, :, 2] = depth_img[i, :]
        points[i, :, 3] = grey_img[i, :]

    # Erase those pixels which are too close or to far to be treated as
    # valuable data.
    UPPER_THRESHOLD = 0.95
    LOWER_THRESHOLD = 0.01
    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            # Replace pixels beyond reasonable value range with median of closest
            # "good" pixels
            if points[i, j, 2] > UPPER_THRESHOLD or points[i, j, 2] < LOWER_THRESHOLD:
                points[i, j, 2] = _median_depth(points, (i,j), IMG_SIZE//3, LOWER_THRESHOLD, UPPER_THRESHOLD)

    # Scale each dimension into interval 0..1
    _normalize(points)

    # Rotate around each axis
    rotation_matrix = np.matmul(_rx(theta_x), np.matmul(_ry(theta_y), _rz(theta_z)))
    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            points[i, j, :3] = np.dot(rotation_matrix, points[i, j, :3].reshape(3,1)).reshape(3)

    # Normalize once more after rotation
    _normalize(points)

    # Apply rotated image to grey and depth photo
    grey_rotated = np.zeros((IMG_SIZE, IMG_SIZE))
    depth_rotated = np.zeros((IMG_SIZE, IMG_SIZE))
    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            x = int(points[i, j, 0] * (IMG_SIZE - 1))
            y = int(points[i, j, 1] * (IMG_SIZE - 1))
            g = points[i, j, 3]
            z = points[i, j, 2]
            grey_rotated[x, y] = g
            depth_rotated[x, y] = z

    grey_rotated = _smoothen(grey_rotated)
    depth_rotated = _smoothen(depth_rotated)

    # If you want to view the rotated image, use the following:
    # tools.show_image(grey_rotated)
    # tools.show_image(depth_rotated)
    # Or:
    # tools.show_3d_plot(X)
    return (grey_rotated, depth_rotated)



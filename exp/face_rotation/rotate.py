import numpy as np
import math
from common.tools import IMG_SIZE
import random
import logging

SMOOTHEN_ITER = 1

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


def _normalize_one_dim(_points):
    """ scales one dimension into interval 0..1 """
    _points -= _points.min()
    if _points.max() > 0:
        _points /= _points.max()

def _normalize(_points):
    """ scales each dimension into interval 0..1 """
    for i in range(_points.shape[-1]):
        _normalize_one_dim(_points[:, :, i])


def _median_neighbors(points, center, size, min_value, max_value):
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
    vals = points[min_x:max_x,min_y:max_y]
    vals = vals[vals >= min_value]
    vals = vals[vals <= max_value]
    if vals.size == 0:
        # Giving up, returning 0 (to be handled in next iter)
        return 0
    return np.median(vals)

def _smoothen(img):
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

def preprocess_images(dimage, image):
    # Erase those pixels which are too close or to far to be treated as
    # valuable data.
    UPPER_THRESHOLD = 0.98
    LOWER_THRESHOLD = 0.1
    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            # Replace pixels beyond reasonable value range with median of closest
            # "good" pixels
            if dimage[i, j] > UPPER_THRESHOLD or dimage[i, j] < LOWER_THRESHOLD:
                new_value = _median_neighbors(dimage[:, :], (i, j), IMG_SIZE // 3, LOWER_THRESHOLD,
                                                    UPPER_THRESHOLD)
                if new_value > UPPER_THRESHOLD or new_value < LOWER_THRESHOLD:
                    new_value = 0.5
                dimage[i, j] = new_value

    # Scale each dimension into interval 0..1
    _normalize_one_dim(dimage)
    _normalize_one_dim(image)


def to_one_matrix(grey_img, depth_img):
    points = np.zeros((IMG_SIZE, IMG_SIZE, 4))
    for i in range(IMG_SIZE):
        points[i, :, 0] = i
        points[:, i, 1] = i
        points[i, :, 2] = depth_img[i, :]
        points[i, :, 3] = grey_img[i, :]
    return points

def rotate_greyd_img(greyd_img, rotation_matrix):
    # First, we prepare the matrix X of points (x, y, z, Grey)
    (grey_img, depth_img) = (greyd_img)
    points = to_one_matrix(grey_img, depth_img)

    # Normalize x an y dimensions of |points|
    _normalize_one_dim(points[:, :, 0])
    _normalize_one_dim(points[:, :, 1])

    preprocess_images(points[:, :, 2], points[:, :, 3])

    # Rotate around each axis
    #rotation_matrix = np.matmul(_rx(theta_x), np.matmul(_ry(theta_y), _rz(theta_z)))
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
            if np.isnan(points[i, j, 0]) or np.isnan(points[i, j, 1]):
                logging.warning("Unexpected NaN in rotated image -- skipping invalid pixel")
                continue
            x = int(points[i, j, 0] * (IMG_SIZE - 1))
            y = int(points[i, j, 1] * (IMG_SIZE - 1))
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
    return (grey_rotated, depth_rotated)



def rotate_greyd_img_by_angle(greyd_img, theta_x=0, theta_y=0, theta_z=0):
    """
    :param greyd_img: a tuple (grey_image, depth_image).
        images are required to be of size tools.IMG_SIZE
    :param theta_x: angle of rotation around axis x
    :param theta_y: angle of rotation around axis y
    :param theta_z: angle of rotation around axis z
    :return: tuple (grey_image, depth_image), rotated by requested angles
    """
    rotation_matrix = np.matmul(_rx(theta_x), np.matmul(_ry(theta_y), _rz(theta_z)))
    return rotate_greyd_img(greyd_img, rotation_matrix)

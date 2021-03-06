import numpy as np
import logging
from scipy import linalg, matrix
from common.constants import FACE_AZIMUTH
import scipy
from model.face import Face


def matrix_null(A: matrix, eps=1e-15):
    u, s, vh = scipy.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)


def calculate_face_normal_vector(x: np.array, y: np.array) -> np.array:
    face_surface = matrix([x, y, [0, 0, 0]])
    v = matrix_null(face_surface)
    return v


def calculate_rotation_beetween_vectors(xy, zy):
    a, b = (xy / np.linalg.norm(xy)).reshape(3), (zy / np.linalg.norm(zy)).reshape(3)
    logging.debug("\nmake " + str(a) + " become " + str(b))
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    I = np.identity(3)
    #logging.debug("crossvectorrotation = " + str(v))
    vXStr = '{} {} {}; {} {} {}; {} {} {}'.format(
        0, -v[2], v[1],
        v[2], 0, -v[0],
        -v[1], v[0], 0
    )
    k = np.matrix(vXStr)
    bb = s ** 2
    if bb == 0:
        return I
    r = I + k + np.dot(k, k) * ((1 - c) / bb)
    logging.debug("ROTATION MATRIX = \n" + str(r))
    return r


def calculate_rotation_matrix(x: tuple, y: tuple, z: tuple) -> tuple:
    xy = np.array(x) - np.array(y)
    zy = np.array(z) - np.array(y)
    xz = np.array(x) - np.array(z)

    v = calculate_face_normal_vector(xy, zy)
    logging.debug("face azimuth = " + str(v))
    rotation = calculate_rotation_beetween_vectors(v, np.array(FACE_AZIMUTH))
    test = np.dot(rotation, v.reshape(3)) - np.array(FACE_AZIMUTH)
    testort = np.dot(xz, v)
    logging.debug("test rot matrix error = " + str(np.linalg.norm(test)))
    logging.debug("test ort error = " + str(np.linalg.norm(testort)))
    return rotation, v


def angle_from(face: Face) -> np.ndarray:
    rotation, azimuth = calculate_rotation_matrix(
        face.face_points["right_brow"],
        face.face_points["top_chin"],
        face.face_points["left_brow"])
    face.azimuth = azimuth
    return rotation


def find_angle(face: Face) -> np.ndarray((3, 3)):
    if len(face.landmarks) > 0:
        rotation = angle_from(face)
        #face.show_position()
        return rotation
    logging.debug("Error, face not found, returning no rotation")
    return None

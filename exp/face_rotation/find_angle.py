import numpy as np
import math
import face_recognition
from common import tools
import numpy.linalg
import common.tools
from math import acos, degrees
from scipy import linalg, matrix
from common.constants import FACE_AZIMUTH
import scipy
from model.face import Face
from face_rotation.face_points import construct_face_points


def matrix_null(A: matrix, eps=1e-15):
    u, s, vh = scipy.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)


def calculate_face_normal_vector(x: np.array, y: np.array) -> np.array:
    face_surface = matrix([x, y, [0, 0, 0]])
    v = matrix_null(face_surface)
    return v


def calculate_rotation_matrix(x: tuple, y: tuple, z: tuple) -> tuple:
    xy = np.array(x) - np.array(y)
    zy = np.array(z) - np.array(y)
    xz = np.array(x) - np.array(z)

    v = calculate_face_normal_vector(xy, zy)
    print("face azimuth = " + str(v))
    rotation = calculate_rotation_beetween_vectors(v, np.array(FACE_AZIMUTH))
    test = np.dot(rotation, v.reshape(3)) - np.array(FACE_AZIMUTH)
    testort = np.dot(xz, v)
    print("test rot matrix error = " + str(np.linalg.norm(test)))
    print("test ort error = " + str(np.linalg.norm(testort)))
    return rotation, v


def calculate_rotation_beetween_vectors(xy, zy):
    a, b = (xy / np.linalg.norm(xy)).reshape(3), (zy / np.linalg.norm(zy)).reshape(3)
    print("\nmake " + str(a) + " become " + str(b))
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    I = np.identity(3)
    #print("crossvectorrotation = " + str(v))
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
    print("ROTATION MATRIX = \n" + str(r))
    return r


def angle_from(face: Face) -> np.ndarray:
    rotation, azimuth = calculate_rotation_matrix(
        face.face_points["right_brow"],
        face.face_points["top_chin"],
        face.face_points["left_brow"])
    face.azimuth = azimuth
    return rotation
def find_angle(face: Face) -> np.ndarray:
    if len(face.landmarks) > 0:
        rotation = angle_from(face)
        face.show_position()
        return rotation
    print("Error, face not found, returning no rotation")
    return None

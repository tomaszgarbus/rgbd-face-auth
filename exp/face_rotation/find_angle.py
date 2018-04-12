import numpy as np
import math
import face_recognition
from common import tools
import numpy.linalg
import common.tools
from math import acos, degrees
from scipy import linalg, matrix
import scipy

FACE_AZIMUTH = np.array([0, 0, 1])


def matrix_null(A, eps=1e-15):
    u, s, vh = scipy.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)


def calculate_face_normal_vector(x, y):
    face_surface = matrix([x, y, [0, 0, 0]])
    v = matrix_null(face_surface)
    return v


def calculate_rotation_matrix(x, y, z):
    xy = np.array(x) - np.array(y)
    zy = np.array(z) - np.array(y)
    xz = np.array(x) - np.array(z)

    v = calculate_face_normal_vector(xy, zy)
    print("face azimuth = " + str(v))
    rotation = calculate_rotation_beetween_vectors(v, FACE_AZIMUTH)
    test = np.dot(rotation, v.reshape(3)) - FACE_AZIMUTH
    testort = np.dot(xz, v)
    print("test rot matrix error = " + str(np.linalg.norm(test)) )
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


def landmarks_take(landmarks):
    for key in list(landmarks.keys()):
        landmarks[key] = list([(x, y) for (y, x) in landmarks[key]])

    avgl = lambda x: tuple([int(sum(y) / len(y)) for y in zip(*x)])
    chin_bottom = landmarks["chin"][6:11]
    chin_left = landmarks["chin"][:6]
    chin_right = landmarks["chin"][11:]
    print("chin_bottom" + str(len(chin_bottom)))
    print("chin_left" + str(len(chin_left)))
    print("chin_right" + str(len(chin_right)))

    right_brow = avgl(landmarks["right_eyebrow"])
    left_brow = avgl(landmarks["left_eyebrow"])
    forehead = avgl([right_brow, left_brow])
    top_chin = avgl(landmarks["bottom_lip"] + chin_bottom + chin_bottom + chin_bottom)
    left_cheek = avgl(chin_left + landmarks["left_eyebrow"] + landmarks["nose_tip"])
    right_cheek = avgl(chin_right + landmarks["right_eyebrow"] + landmarks["nose_tip"])
    face_center = avgl([right_brow] + [left_brow] + [top_chin])

    return {"right_brow": [right_brow], "left_brow": [left_brow], "forehead": [forehead],
            "top_chin": [top_chin]}


def angle_from(landmarks, imaged, shape):
    get_pixl = lambda x0, x1: imaged[max(0, min(shape[1]-1, x0)), max(0, min(shape[0]-1, x1))]
    to3d = lambda x: (x[0]/shape[0], x[1]/shape[1], get_pixl(x[0], x[1]))

    right_brow = to3d(landmarks["right_brow"][0])
    left_brow = to3d(landmarks["left_brow"][0])
    top_chin = to3d(landmarks["top_chin"][0])
    forehead = to3d(landmarks["forehead"][0])

    print(str({"right_brow": [right_brow], "left_brow": [left_brow],  "top_chin": [top_chin], "forehead": [forehead]}))

    rotation, azimuth = calculate_rotation_matrix(right_brow, top_chin, left_brow)

    return rotation, forehead, azimuth


def show_with_landmarks(image, landmarks, azimuth, face_center):
    img = np.copy(image)
    mxx = img.shape[0] - 1
    mxy = img.shape[1] - 1
    print(str(landmarks))
    for key in landmarks.keys():
        for (x, y) in landmarks[key]:
            img[min(max(x, 0), mxx), min(max(y, 0), mxy)] = 1

    v = np.array([face_center[0]*(mxx + 1), face_center[1]*(mxy+1), face_center[2]])
    azimuth = azimuth[0]
    for i in range(100):
        x = min(max(int(v[0]), 0), mxx)
        y = min(max(int(v[1]), 0), mxy)
        img[x,y] = ((100-i)/100)
        if i % 10 == 0:
            print("point on " + str(x) + "," +str(y))
        v -= azimuth

    tools.show_image(img)


def load_face_points(image):
    img1 = (image*256).astype(np.uint8)
    return face_recognition.face_landmarks(img1)


def find_angle(image, imaged):
    print("\n\nFIND ANGLE")
    face_points = load_face_points(image)
    if len(face_points) > 0:
        landmarks = landmarks_take(face_points[0])
        rotation, face_center, azimuth = angle_from(landmarks, imaged, image.shape)
        show_with_landmarks(image, landmarks, azimuth, face_center)
        center = np.dot(rotation, np.array(face_center)).reshape(3)
        print("center = " +str(center))
        return rotation, (center.item(1), center.item(0))
    print("Error, face not found, returning no rotation")
    return None, (1/2, 1/5)

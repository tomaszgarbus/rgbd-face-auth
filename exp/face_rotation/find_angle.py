import numpy as np
import math
import face_recognition
from common import tools
import numpy.linalg
import common.tools
from math import acos, degrees


def get_angle(x, y, z):

    print("Wspolrzedne:")
    print("x:" + str(x))
    print("y:" + str(y))
    print("z:" + str(z))

    yx = np.linalg.norm(np.array(x) - np.array(y))
    yz = np.linalg.norm(np.array(z) - np.array(y))
    xz = np.linalg.norm(np.array(x) - np.array(z))

    r = acos((yx * yx + yz * yz - xz * xz) / (2.0 * yx * yz))

    print("Trojkat: " + str(yx) + "," + str(yz) + "," + str(xz))
    print("Kat: " + str(degrees(r)))

    return r


def angle_to_align_depth(x, y):
    return get_angle(x, y, (x[0], x[1], y[2])) * (-1 if x[2] > y[2] else 1)


def angle_to_align_height(x, y):
    return get_angle(x, y, (x[0], y[1], x[2])) * (-1 if x[1] < y[1] else 1)


def landmarks_take(landmarks):
    avgl = lambda x: tuple([int(sum(y) / len(y)) for y in zip(*x)])
    chin_bottom = landmarks["chin"][7:11]
    chin_left = landmarks["chin"][:7]
    chin_right = landmarks["chin"][12:]

    right_brow = avgl(landmarks["right_eyebrow"])
    left_brow = avgl(landmarks["left_eyebrow"])
    forehead = avgl([right_brow, left_brow])
    top_chin = avgl(landmarks["bottom_lip"] + chin_bottom + chin_bottom)
    left_cheek = avgl(chin_left + landmarks["left_eyebrow"] + landmarks["nose_tip"])
    right_cheek = avgl(chin_right + landmarks["right_eyebrow"] + landmarks["nose_tip"])

    return {"right_brow": [right_brow], "left_brow": [left_brow], "forehead": [forehead], "top_chin": [top_chin]}


def angle_from(landmarks, imaged, shape):
    to3d = lambda x: (x[0]/shape[0], x[1]/shape[1], imaged[x[1], x[0]])

    right_brow = to3d(landmarks["right_brow"][0])
    left_brow = to3d(landmarks["left_brow"][0])
    forehead = to3d(landmarks["forehead"][0])
    top_chin = to3d(landmarks["top_chin"][0])

    print(str({"right_brow": [right_brow], "left_brow": [left_brow], "forehead": [forehead], "top_chin": [top_chin]}))

    print("boki")
    x = angle_to_align_depth(right_brow, left_brow)
    print("pion")
    y = angle_to_align_depth(forehead, top_chin)
    print("obrot")
    z = angle_to_align_height(right_brow, left_brow)
    print("angle " + str(x) + " " + str(y) + " " + str(z))

    print("face rotated " + ("right" if x > 0 else "left") + " and " + ("down" if y > 0 else "up"))

    return x, y, z, forehead


def show_with_landmarks(image, landmarks):
    img = np.copy(image)
    mxx = img.shape[0] - 1
    mxy = img.shape[1] - 1
    print(str(landmarks))
    for key in landmarks.keys():
        for (x, y) in landmarks[key]:
            img[min(max(y, 0), mxx), min(max(x, 0), mxy)] = 1
    tools.show_image(img)


def get_landmarks(image):
    img1 = (image*256).astype(np.uint8)
    return face_recognition.face_landmarks(img1)


def find_angle(image, imaged):
    print("\n\nFIND ANGLE")
    face_landmarks_list = get_landmarks(image)
    if len(face_landmarks_list) > 0:
        landmarks = landmarks_take(face_landmarks_list[0])
        # show_with_landmarks(image, face_landmarks_list[0])
        show_with_landmarks(image, landmarks)
        return angle_from(landmarks, imaged, image.shape)
    print("Error, face not found, returning no rotation")
    return 0, 0, 0, (1/2, 1/5)

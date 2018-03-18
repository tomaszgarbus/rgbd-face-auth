import numpy as np
import math
import face_recognition
from common import tools
import common.tools


def get_angle(x, y, z):
    yx = np.array(x) - np.array(y)
    yz = np.array(z) - np.array(y)

    cosine_angle = np.dot(yx, yz) / (np.linalg.norm(yx) * np.linalg.norm(yz))
    angle = np.arccos(cosine_angle)

    return angle * (-1 if x[2] > y[2] else 1)


def angle_to_align(x, y):
    return get_angle(x, y, (x[0], x[1], y[2]))


def landmarks_take(landmarks):
    avgl = lambda x: tuple([int(sum(y) / len(y)) for y in zip(*x)])
    chin_bottom = landmarks["chin"][7:11]
    chin_left = landmarks["chin"][:7]
    chin_right = landmarks["chin"][12:]

    rightbrew = avgl(landmarks["right_eyebrow"])
    leftbrew = avgl(landmarks["left_eyebrow"])
    forhead = avgl([rightbrew, leftbrew])
    topchin = avgl(landmarks["bottom_lip"] + chin_bottom + chin_bottom)
    left_cheek = avgl(chin_left + landmarks["left_eyebrow"] + landmarks["nose_tip"])
    right_cheek = avgl(chin_right + landmarks["right_eyebrow"] + landmarks["nose_tip"])

    return {"rightbrew": [rightbrew], "leftbrew": [leftbrew], "forhead": [forhead], "topchin": [topchin]}


def angle_from(landmarks, imaged, shape):
    to3d = lambda x: (x[0]/shape[0], x[1]/shape[1], imaged[x[1], x[0]])

    rightbrew = to3d(landmarks["rightbrew"][0])
    leftbrew = to3d(landmarks["leftbrew"][0])
    forhead = to3d(landmarks["forhead"][0])
    topchin = to3d(landmarks["topchin"][0])

    print(str({"rightbrew": [rightbrew], "leftbrew": [leftbrew], "forhead": [forhead], "topchin": [topchin]}))

    x = angle_to_align(rightbrew, leftbrew)
    y = angle_to_align(forhead, topchin)
    z = 0
    print("angle " + str(x) + " " + str(y) + " " + str(z))

    print("face rotated " + ("right" if x > 0 else "left") + " and " + ("down" if y > 0 else "up"))

    return x, y, z


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
    face_landmarks_list = get_landmarks(image)
    if len(face_landmarks_list) > 0:
        landmarks = landmarks_take(face_landmarks_list[0])
        # show_with_landmarks(image, face_landmarks_list[0])
        show_with_landmarks(image, landmarks)
        return angle_from(landmarks, imaged, image.shape)
    print("Error, face not found, returning no rotation")
    return 0, 0, 0

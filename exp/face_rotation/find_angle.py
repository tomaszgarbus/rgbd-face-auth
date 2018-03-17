import numpy as np
import math
import face_recognition
from common import tools
import common.tools


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

    return {"all": [rightbrew, leftbrew, forhead, topchin]}  # , eft_cheek, right_cheek]}


def angle_from(landmarks):
    return None

def show_with_landmarks(image, landmarks):
    img = np.copy(image[:, :, 3])
    mxx = img.shape[0] - 1
    mxy = img.shape[1] - 1
    print(str(landmarks))
    for key in landmarks.keys():
        for (x, y) in landmarks[key]:
            img[min(max(y, 0), mxx), min(max(x, 0), mxy)] = 1
    tools.show_image(img)


def get_landmarks(image):
    img1 = (image[:, :, 3]*256).astype(np.uint8)
    return face_recognition.face_landmarks(img1)


def find_angle(image):
    face_landmarks_list = get_landmarks(image)
    if len(face_landmarks_list) > 0:
        landmarks = landmarks_take(face_landmarks_list[0])
        show_with_landmarks(image, face_landmarks_list[0])
        show_with_landmarks(image, landmarks)
        return angle_from(landmarks)
    return None

import numpy as np
import math
import face_recognition
from common import tools
import common.tools


def landmarks_compress(landmarks):
    pass

def show_with_landmarks(image, landmarks):
    img = image[:, :, 3]
    mxx, mxy = image.shape
    for key in landmarks.keys():
        for (x, y) in landmarks[key]:
            img[min(max(x, 0), mxx), min(max(y, 0), mxy)] = 1
    tools.show_image(img)


def get_landmarks(image):
    img1 = (image[:, :, 3]*256).astype(np.uint8)
    return face_recognition.face_landmarks(img1)


def find_angle(image):
    face_landmarks_list = get_landmarks(image)
    if len(face_landmarks_list) > 0:
        show_with_landmarks(image, face_landmarks_list[0])
        #show_with_landmarks(image, landmarks_compress(face_landmarks_list[0]))

    print(str(face_landmarks_list))
    return img2
    
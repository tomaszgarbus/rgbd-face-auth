import numpy as np
from common import tools
from common.tools import IMG_SIZE

CENTER_DEST = (1/2, 1/5) # where center should be

def show_with_center(image, center):
    img = np.copy(image)
    x = int(CENTER_DEST[0] * IMG_SIZE)
    y = int(CENTER_DEST[1] * IMG_SIZE)
    img[y, x] = 1
    tools.show_image(img)


def get_landmarks(image):
    img1 = (image*256).astype(np.uint8)
    return face_recognition.face_landmarks(img1)


def recentre(image, imaged, center):
    assert image.shape == imaged.shape
    print("\n\nRECENTRE")
    move_x = int((CENTER_DEST[0] - center[0]) * IMG_SIZE)
    move_y = int((CENTER_DEST[1] - center[1]) * IMG_SIZE)

    image = np.roll(image, move_x, axis=1)
    image = np.roll(image, move_y, axis=0)
    imaged = np.roll(imaged, move_x, axis=1)
    imaged = np.roll(imaged, move_y, axis=0)
    show_with_center(imaged, center)

    return image, imaged




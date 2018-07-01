import numpy as np
from common import tools
from common.constants import IMG_SIZE, CENTER_DEST
from model.face import Face
import logging


def show_with_center(image, center):
    img = np.copy(image)
    x = int(center[0] * IMG_SIZE)
    y = int(center[1] * IMG_SIZE)
    img[y, x] = 1
    tools.show_image(img)


def recentre(face: Face) -> None:
    assert face.depth_img.shape == face.gir_img.shape
    logging.debug("\n\nRECENTRE")
    move_x = int((CENTER_DEST[0] - face.face_center[0]) * IMG_SIZE)
    move_y = int((CENTER_DEST[1] - face.face_center[1]) * IMG_SIZE)

    logging.debug("MOVE X MOVE Y %d %d" % (move_x, move_y))
    face.gir_img = np.roll(face.gir_img, move_x, axis=1)
    face.gir_img = np.roll(face.gir_img, move_y, axis=0)
    face.depth_img = np.roll(face.depth_img, move_x, axis=1)
    face.depth_img = np.roll(face.depth_img, move_y, axis=0)
    if move_x >= 0:
        face.gir_img[:, move_x] = 0
        face.depth_img[:, move_x] = 0
    else:
        face.gir_img[:, move_x:] = 0
        face.depth_img[:, move_x:] = 0
    if move_y >= 0:
        face.gir_img[:move_y, :] = 0
        face.depth_img[:move_y, :] = 0
    else:
        face.gir_img[move_y:, :] = 0
        face.depth_img[move_y:, :] = 0
    #show_with_center(imaged, center)


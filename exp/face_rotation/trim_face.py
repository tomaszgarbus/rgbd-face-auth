import face_recognition
import numpy as np
import logging
from scipy.spatial import ConvexHull
import sys
from common import tools
from common.constants import IMG_SIZE, BGCOLOR
from model.face import Face

def find_convex_hull_vertices(grey_img: np.ndarray) -> list:
    """
        :param grey_img:
        :return: A list of 2-D points denoting positions of vertices
            of convex hull on |grey_img|
    """
    landmarks = face_recognition.face_landmarks((grey_img * 256).astype(np.uint8))
    if landmarks == []:
        logging.warning("Face not found")
        return []
    landmarks = landmarks[0]
    all_points = []
    for key in landmarks:
        for point in landmarks[key]:
            x, y = point
            try:
                all_points.append((y, x))
            except IndexError:
                logging.warning("Pixel at (%d, %d) out of image" % (x, y))
    hull = ConvexHull(all_points)
    ch_vertices = hull.points[hull.vertices]
    return ch_vertices


def connect_convex_hull_vertices(depth_img: np.ndarray, ch_vertices: list) -> list:
    all_points = []

    for point1, point2 in zip(ch_vertices, np.roll(ch_vertices, 1, axis=0)):
        xs, ys = point1
        xe, ye = point2
        difx = xe - xs
        dify = ye - ys
        steps = int(max(abs(difx), abs(dify)))
        stepx = difx / steps
        stepy = dify / steps
        for i in range(steps):
            try:
                all_points.append((int(xs), int(ys), depth_img[int(xs), int(ys)]))
            except IndexError:
                #logging.warning("Pixel out of image")
                pass
            xs += stepx
            ys += stepy

    return all_points


def find_convex_hull(face: Face) -> list:
    """
        Finds all points lying on the convex hull of the face. Applies
        depth to each point so that it can be rotated the same way as the face
        :param face:
        :return: A list of 3-D points (x,y,depth)
    """
    ch_vertices = find_convex_hull_vertices(face.grey_img)
    if ch_vertices == []:
        logging.warning("Face was not found")
        return []

    return connect_convex_hull_vertices(face.depth_img, ch_vertices)


def generate_mask(face: Face, points: list((float, float, float))) -> None:
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.bool)
    for (xs, ys, _) in points:
        mask[int(xs), int(ys)] = True
    starting_point = (IMG_SIZE//2, IMG_SIZE//2)  # TODO: maybe some specific landmark (like nose)

    sys.setrecursionlimit(100000)

    def _helper(px, py):
        if min(px, py) < 0 or max(px, py) >= IMG_SIZE:
            return
        if mask[px, py]:
            return
        mask[px, py] = True
        _helper(px-1, py)
        _helper(px+1, py)
        _helper(px, py-1)
        _helper(px, py+1)
    _helper(starting_point[0], starting_point[1])
    face.mask = mask

    # Display the mask if you want
    # tools.show_image(mask)


def generate_mask_from_skin(face: Face) -> None:
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.bool)
    for x in range(IMG_SIZE):
        for y in range(IMG_SIZE):
            r, g, b = face.rgb_img[x][y]
            if tools.rgb_skin_check(r, g, b):
                mask[x][y] = True

    face.mask = mask

    # Display the mask if you want
    tools.show_image(face.rgb_img)
    tools.show_image(mask)


def cut_around_mask(face: Face,
                    color: float = BGCOLOR) -> None:
    """
        Erases contents of original image around mask.
        :param face:
        :param color to fill around mask
    """
    for x in range(IMG_SIZE):
        for y in range(IMG_SIZE):
            if not face.mask[x, y]:
                face.depth_img[x, y] = color
                face.grey_img[x, y] = color


def trim_greyd(face: Face) -> None:
    """
        :param face
        :return: trimmed Face
    """

    # Approach 1: use face_recoginition library (better but slower)
    all_points = find_convex_hull(face)
    generate_mask(face, all_points)

    # Approach 2: use heuristics for finding skin pixels
    # generate_mask_from_skin(face)
    # (just comment out the one you don't like)

    cut_around_mask(face)

    #tools.show_image(grey_img)
    #tools.show_image(depth_img)

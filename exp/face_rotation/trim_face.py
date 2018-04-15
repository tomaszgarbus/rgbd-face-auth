import face_recognition
import numpy as np
import logging
from scipy.spatial import ConvexHull
import sys
from common import tools
from common.constants import IMG_SIZE, BGCOLOR
from model.face import Face


def paint_bucket(image: np.ndarray, posx: int = 0, posy: int = 0, color: float = BGCOLOR) -> np.ndarray:
    """
        Fills image with color |color| if pixel has different color and
        propagates to neighboring pixels
    """
    def _helper(px, py):
        if px < 0 or py < 0 or px >= image.shape[0] or py >= image.shape[1]:
            return
        if image[px, py] == color:
            return
        image[px, py] = color
        _helper(px-1, py)
        _helper(px+1, py)
        _helper(px, py-1)
        _helper(px, py+1)
    _helper(posx, posy)
    return image


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
            y, x = point
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
        ys, xs = point1
        ye, xe = point2
        difx = xe - xs
        dify = ye - ys
        steps = int(max(abs(difx), abs(dify)))
        stepx = difx / steps
        stepy = dify / steps
        for i in range(steps):
            try:
                all_points.append((int(xs), int(ys), depth_img[int(xs), int(ys)]))
            except IndexError:
                logging.warning("Pixel out of image")
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
        return []

    return connect_convex_hull_vertices(face.depth_img, ch_vertices)


def generate_mask(face: Face, points: list((float, float, float))) -> None:
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.bool)
    for (xs, ys, _) in points:
        mask[int(xs), int(ys)] = True
    starting_point = (IMG_SIZE//2, IMG_SIZE//2)  # TODO: maybe some specific landmark (like nose)

    def _helper(px, py):
        if mask(px, py):
            return
        if min(px, py) < 0 or max(px, py) >= IMG_SIZE:
            return
        mask[px, py] = True
        _helper(px-1, py)
        _helper(px+1, py)
        _helper(px, py-1)
        _helper(px, py+1)
    _helper(starting_point[0], starting_point[1])
    face.mask = mask
    tools.show_image(mask)

def cut_around_points(face, points, color=BGCOLOR):
    """
        Erases contents of original image around |points|.
        :param face:
        :param points: list of 3D points, must be isolating a face perfectly
        :return: modified grey_img, depth_img
    """
    for (xs, ys, _) in points:
        face.depth_img[int(xs), int(ys)] = color
        face.grey_img[int(xs), int(ys)] = color

    sys.setrecursionlimit(100000)
    for posx in [0,IMG_SIZE-1]:
        for posy in [0,IMG_SIZE-1]:
            grey_img = paint_bucket(face.grey_img, posx=posx, posy=posy, color=color)
            depth_img = paint_bucket(face.depth_img, posx=posx, posy=posy, color=color)
    return grey_img, depth_img


def trim_greyd(face: Face) -> (Face, list):
    """
        :param face
        :return: (trimmed Face, list of 2D points)
    """
    grey_img, depth_img = face
    ch_vertices = find_convex_hull_vertices(grey_img)

    if ch_vertices == []:
        return face, []

    all_points = find_convex_hull(face)

    grey_img, depth_img = cut_around_points(face, all_points)

    #tools.show_image(grey_img)
    #tools.show_image(depth_img)

    return Face(grey_img, depth_img), ch_vertices
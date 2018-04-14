import face_recognition
import numpy as np
from common import tools
from common.tools import IMG_SIZE
import logging
from scipy.spatial import ConvexHull
import sys

BGCOLOR = 0


def paint_bucket(image, posx=0, posy=0, color=BGCOLOR):
    """Fills image with color |color| if pixel has different color and
    propagates to neighboring pixels"""
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


def find_convex_hull(grey_img):
    landmarks = face_recognition.face_landmarks((grey_img * 256).astype(np.uint8))
    if landmarks == []:
        logging.warning("Face not found")
        return []
    landmarks = landmarks[0]
    all_points = []
    for key in landmarks:
        for point in landmarks[key]:
            y, x = point
            all_points.append((y, x))
    hull = ConvexHull(all_points)
    ch_vertices = hull.points[hull.vertices]
    return ch_vertices


def trim_greyd(grey_img, depth_img):
    ch_vertices = find_convex_hull(grey_img)

    if ch_vertices == []:
        return (grey_img, depth_img)

    for point1, point2 in zip(ch_vertices, np.roll(ch_vertices, 1, axis=0)):
        ys,xs = point1
        ye,xe = point2
        difx = xe-xs
        dify = ye-ys
        steps = int(max(abs(difx), abs(dify)))
        stepx = difx/steps
        stepy = dify/steps
        for i in range(steps):
            try:
                depth_img[int(xs), int(ys)] = BGCOLOR
                grey_img[int(xs), int(ys)] = BGCOLOR
            except IndexError:
                logging.warning("Pixel out of image")
            xs += stepx
            ys += stepy

    sys.setrecursionlimit(100000)
    for posx in [0,IMG_SIZE-1]:
        for posy in [0,IMG_SIZE-1]:
            grey_img = paint_bucket(grey_img, posx=posx, posy=posy)
            depth_img = paint_bucket(depth_img, posx=posx, posy=posy)

    #tools.show_image(grey_img)
    #tools.show_image(depth_img)

    return (grey_img, depth_img)
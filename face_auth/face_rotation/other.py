from model.face import Face
import numpy as np

def construct_face_points(face: Face) -> None:
    shape = face.grey_img.shape
    landmarks = face.landmarks

    avgl = lambda x: tuple([int(sum(y) / len(y)) for y in zip(*x)])
    get_pixl = lambda x0, x1: face.depth_img[max(0, min(shape[0] - 1, x0)), max(0, min(shape[1] - 1, x1))]
    to3d = lambda x: (x[0] / shape[0], x[1] / shape[1], get_pixl(x[0], x[1]))

    def avgl3d(l: list((int, int))) -> np.ndarray((1, 3)):
        # Map 2D points to 3D by applying depth
        points3d = np.array([(x / shape[0], y / shape[1], get_pixl(x, y)) for (x, y) in l])
        # Filter out the points too far from mean depth
        me = np.mean(points3d[:, 2])
        stdev = np.std(points3d[:, 2])
        depths_good = np.abs(points3d[:, 2] - me) <= stdev
        points3d = points3d[depths_good]
        return points3d.sum(axis=0) / points3d.shape[0]

    chin_bottom = landmarks["chin"][6:11]
    # chin_left = landmarks["chin"][:6]
    # chin_right = landmarks["chin"][11:]

    right_brow = avgl(landmarks["right_eyebrow"])
    left_brow = avgl(landmarks["left_eyebrow"])
    forehead = avgl([right_brow, left_brow])
    # top_chin = avgl(landmarks["bottom_lip"] + chin_bottom + chin_bottom + chin_bottom)
    # left_cheek = avgl(chin_left + landmarks["left_eyebrow"] + landmarks["nose_tip"])
    # right_cheek = avgl(chin_right + landmarks["right_eyebrow"] + landmarks["nose_tip"])
    # face_center = avgl([right_brow] + [left_brow] + [top_chin])
    right_brow3d = avgl3d(landmarks["right_eyebrow"])
    left_brow3d = avgl3d(landmarks["left_eyebrow"])
    top_chin3d = avgl3d(landmarks["bottom_lip"] + chin_bottom + chin_bottom + chin_bottom)

    face.face_center = to3d(forehead)
    face.face_points = {"right_brow": right_brow3d, "left_brow": left_brow3d, "top_chin": top_chin3d}


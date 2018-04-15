from model.face import Face
from face_rotation.rotate import _rescale_one_dim
from common.constants import IMG_SIZE


def construct_face_points(face: Face) -> None:
    shape = face.grey_img.shape
    landmarks = face.landmarks

    avgl = lambda x: tuple([int(sum(y) / len(y)) for y in zip(*x)])
    get_pixl = lambda x0, x1: face.depth_img[max(0, min(shape[0] - 1, x0)), max(0, min(shape[1] - 1, x1))]
    to3d = lambda x: (x[0] / shape[0], x[1] / shape[1], get_pixl(x[0], x[1]))

    chin_bottom = landmarks["chin"][6:11]
    chin_left = landmarks["chin"][:6]
    chin_right = landmarks["chin"][11:]

    right_brow = avgl(landmarks["right_eyebrow"])
    left_brow = avgl(landmarks["left_eyebrow"])
    forehead = avgl([right_brow, left_brow])
    top_chin = avgl(landmarks["bottom_lip"] + chin_bottom + chin_bottom + chin_bottom)
    left_cheek = avgl(chin_left + landmarks["left_eyebrow"] + landmarks["nose_tip"])
    right_cheek = avgl(chin_right + landmarks["right_eyebrow"] + landmarks["nose_tip"])
    face_center = avgl([right_brow] + [left_brow] + [top_chin])

    face.face_center = to3d(forehead)
    face.face_points = {"right_brow": to3d(right_brow), "left_brow": to3d(left_brow), "top_chin": to3d(top_chin)}


def drop_corner_values(face: Face,
                       lower_threshold : float = 0.1,
                       upper_threshold : float = 0.98) -> None:
    """
        Erase those pixels which are too close or to far to be treated as
        valuable data.
    """
    depth_median = np.median(face.depth_img[face.mask])
    face.depth_img[face.mask] -= depth_median
    depth_stdev = np.std(face.depth_img[face.mask])
    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            if abs(face.depth_img[i, j]) >= depth_stdev:
                face.depth_img[i, j] = 0

    face.depth_img /= depth_stdev


    # Scale each dimension into interval 0..1
    _rescale_one_dim(face.depth_img)
    _rescale_one_dim(face.grey_img)


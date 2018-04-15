from model.face import Face


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


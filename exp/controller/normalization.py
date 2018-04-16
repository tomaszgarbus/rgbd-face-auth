import numpy as np
import face_recognition

from model.face import Face
from face_rotation.trim_face import trim_greyd
from face_rotation.rotate import rotate_greyd_img, drop_corner_values
from face_rotation.other import construct_face_points
from face_rotation.find_angle import find_angle
from face_rotation.recentre import recentre


def preprocessing(face: Face) -> None:
    if face.preprocessed:
        return
    face.preprocessed = True

    # Display the original photo
    face.show_grey()
    face.show_depth()

    # Trim face
    trim_greyd(face)

    # Display trimmed photo
    face.show_grey()
    face.show_depth()

    # Drop corner values and rescale to 0...1
    drop_corner_values(face)

    # Calculate face center and points defining face surface
    construct_face_points(face)

    # Display the photo after normalizing mean
    face.show_position()
    face.show_depth()


def normalized(face: Face) -> Face:
    preprocessing(face)

    # Find the angle
    rotation = find_angle(face)
    if rotation is None:
        return None

    # Apply rotation
    rotated_face = rotate_greyd_img(face, rotation)

    # Display the results
    rotated_face.show_position()
    rotated_face.show_depth()

    # centering
    rotated_face.show_position()
    recentre(rotated_face)
    rotated_face.show_position()

    # tools.show_3d_plot(rotate.to_one_matrix(rotated_face))



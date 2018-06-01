import numpy as np
import face_recognition

from model.face import Face
from face_rotation.trim_face import trim_greyd
from face_rotation.rotate import rotate_greyd_img, drop_corner_values
from face_rotation.other import construct_face_points
from face_rotation.find_angle import find_angle
from face_rotation.recentre import recentre
from face_rotation.structure_filters import get_hog_of, get_entropy_map_of


def preprocessing(face: Face,
                  trim_method: str = 'convex_hull') -> None:
    if face.preprocessed:
        return
    face.preprocessed = True

    # Display the original photo
    #face.show_grey()
    #face.show_depth()

    # Trim face
    trim_greyd(face, method=trim_method)

    # Display trimmed photo
    #face.show_grey()
    #face.show_depth()

    # Drop corner values and rescale to 0...1
    drop_corner_values(face)

    # Calculate face center and points defining face surface
    construct_face_points(face)

    # Display the photo after normalizing mean
    #face.show_position()
    #face.show_depth()


def normalized(face: Face,
               rotate: bool = True,
               trim_method: str = 'convex_hull') -> Face:
    preprocessing(face, trim_method=trim_method)

    if not rotate:
        return face

    # Find the angle
    rotation = find_angle(face)
    if rotation is None:
        return None

    # Apply rotation
    rotated_face = rotate_greyd_img(face, rotation)

    # Display the results
    #rotated_face.show_position()
    #rotated_face.show_depth()

    # centering
    #rotated_face.show_position()
    recentre(face)
    #rotated_face.show_position()



    return rotated_face
    # tools.show_3d_plot(rotate.to_one_matrix(rotated_face))


def hog_and_entropy(face: Face) -> Face:
    face.hog_grey_image, fdg = get_hog_of(face.grey_img)
    face.hog_grey_fd = fdg
    face.entropy_map_grey_image = get_entropy_map_of(face.grey_img)
    face.hog_depth_image, fdd = get_hog_of(face.depth_img)
    face.hog_depth_fd = fdd
    face.entropy_map_depth_image = get_entropy_map_of(face.depth_img)
    return face

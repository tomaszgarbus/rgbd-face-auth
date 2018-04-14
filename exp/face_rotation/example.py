""" A simple demo to demonstrate an example usage of module rotate. """


from common.db_helper import DBHelper, Database
from face_rotation import rotate
from common import tools
from face_rotation.find_angle import find_angle
from face_rotation.recentre import recentre, show_with_center
from face_rotation import trim_face
import face_rotation.find_angle
import numpy as np

if __name__ == '__main__':
    def load_samples(database, limit=10):
        samples = []
        print('Loading database %s with limit %d' % (database.get_name(), limit))
        for i in range(database.subjects_count()):
            for j in range(database.imgs_per_subject(i)):
                if len(samples) >= limit:
                    return samples
                x = database.load_greyd_face(i, j)
                if x[0] is None or x[1] is None:
                    continue
                samples.append(x)
        return samples

    # Load a random photo to rotate
    helper = DBHelper()
    TOTAL_SUBJECTS_COUNT = helper.all_subjects_count()
    photos = []
    for database in helper.get_databases():
        photos += load_samples(database, limit=2)

    for img_grey, img_depth in photos:

        # Trim face
        img_grey, img_depth, convex_hull_vertices = trim_face.trim_greyd(img_grey, img_depth)

        rotate.preprocess_images(img_depth, img_grey)

        # Display the photo before rotation
        #tools.show_image(img_grey)
        #tools.show_image(img_depth)

        # find the angle
        rotation, center, face_points = find_angle(img_grey, img_depth)

        if rotation is None :
            continue
        print("center = " + str(center))

        # Apply rotation
        rotated_grey, rotated_depth = rotate.rotate_greyd_img((img_grey, img_depth), rotation)
        face_rotation.find_angle.show_with_landmarks_zeroone(rotated_grey, face_points)

        #show_with_center(rotated_grey, center)
        rotated_grey, rotated_depth = recentre(rotated_grey, rotated_depth, center)
        show_with_center(rotated_grey, (1/2, 1/5))

        #tools.show_3d_plot(rotate.to_one_matrix(rotated_grey, rotated_depth))
        # Display the results
        tools.show_image(rotated_depth)
        tools.show_image(rotated_grey)

        # exit(0)


""" A simple demo to demonstrate an example usage of module rotate. """


from common.db_helper import DBHelper, Database
from face_rotation import rotate
from common import tools
from face_rotation.find_angle import find_angle

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
        if database.get_name() == 'ias_lab_rgbd':
            photos += load_samples(database, limit=1)

    for img_grey, img_depth in photos:

        rotate.preprocess_images(img_depth, img_grey)

        # Display the photo before rotation
        #tools.show_image(img_grey)
        #tools.show_image(img_depth)

        # find the angle
        theta_x, theta_y, theta_z = find_angle(img_grey, img_depth)

        # Apply rotation
        rotated_grey, rotated_depth = rotate.rotate_greyd_img((img_grey, img_depth),
                                                              theta_x=theta_x,
                                                              theta_y=theta_y,
                                                              theta_z=theta_z)

        tools.show_3d_plot(rotate.to_one_matrix(rotated_grey, rotated_depth))
        # Display the results
        tools.show_image(rotated_grey)
        #tools.show_image(rotated_depth)



""" A simple demo to demonstrate an example usage of module rotate. """


from common.db_helper import DBHelper, Database
from face_rotation import rotate
from common import tools

if __name__ == '__main__':
    def test_samples(database, limit=10000):
        samples = []
        print('Loading database %s with limit %d' % (database.get_name(), limit))
        for i in range(database.subjects_count()):
            for j in range(database.imgs_per_subject(i)):
                print(i, j)
                if len(samples) >= limit:
                    return samples
                x = database.load_greyd_face(i, j)
                if x[0] is None or x[1] is None:
                    continue
                samples.append(x)
                img_grey, img_depth = x
                rotated_grey, rotated_depth = rotate.rotate_greyd_img((img_grey, img_depth),
                                                                      theta_x=0.2,
                                                                      theta_y=0.0,
                                                                      theta_z=0.0)
                tools.show_image(img_grey)
                tools.show_image(rotated_grey)
        return samples

    # Load a random photo to rotate
    helper = DBHelper()
    TOTAL_SUBJECTS_COUNT = helper.all_subjects_count()
    for database in helper.get_databases():
        if database.get_name() == 'www.vap.aau.dk':
            test_samples(database)


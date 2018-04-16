""" A simple demo to demonstrate an example usage of module rotate. """


from common.db_helper import DBHelper
from controller.normalization import normalized


if __name__ == '__main__':
    def load_samples(database, limit=10):
        samples = []
        print('Loading database %s with limit %d' % (database.get_name(), limit))
        for i in range(database.subjects_count()):
            for j in range(database.imgs_per_subject(i)):
                if len(samples) >= limit:
                    return samples
                x = database.load_greyd_face(i, j)
                if x.grey_img is None or x.depth_img is None:
                    continue
                samples.append(x)
        return samples

    # Load a random photo to rotate
    helper = DBHelper()
    TOTAL_SUBJECTS_COUNT = helper.all_subjects_count()
    photos = []
    for database in helper.get_databases():
        if database.get_name() != 'www.vap.aau.dk':
            photos += load_samples(database, limit=1)

    for face in photos[:15]:
        face = normalized(face)
        face.show_grey()
        face.show_depth()




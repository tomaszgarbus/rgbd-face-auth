""" A simple demo to demonstrate an example usage of module rotate. """


from common.db_helper import DBHelper
from controller.normalization import normalized, hog_and_entropy
import logging
from common.tools import show_image


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    def load_samples(database, limit=10, limit_per_person=2):
        samples = []
        logging.info('Loading database %s with limit %d' % (database.get_name(), limit))
        for i in range(database.subjects_count()):
            person_photos_count = 0
            for j in range(database.imgs_per_subject(i)):
                if len(samples) >= limit:
                    return samples
                x = database.load_greyd_face(i, j)
                if x.grey_img is None or x.depth_img is None:
                    continue
                samples.append(x)
                person_photos_count += 1
                if limit_per_person is not None\
                   and person_photos_count >= limit_per_person:
                    break
        return samples

    # Load a random photo to rotate
    helper = DBHelper()
    TOTAL_SUBJECTS_COUNT = helper.all_subjects_count()
    photos = []
    for database in helper.get_databases():
        if database.get_name() != 'www.vap.aau.dk':
            photos += load_samples(database, limit=3, limit_per_person=3)

    for face in photos:
        face = normalized(face)
        face = hog_and_entropy(face)

        inp = face.get_concat()
        show_image(inp)




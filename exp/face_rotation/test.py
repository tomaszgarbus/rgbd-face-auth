from common import tools
from common.tools import IMG_SIZE
from common.db_helper import DBHelper, Database
from face_rotation import rotate
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
                if x is None:
                    continue
                samples.append(x)
        return samples


    helper = DBHelper()
    TOTAL_SUBJECTS_COUNT = helper.all_subjects_count()
    photos = []
    for database in helper.get_databases():
        photos += load_samples(database, limit=1)


    X = np.zeros((IMG_SIZE, IMG_SIZE, 4))
    for i in range(64):
        X[i, :, 0] = i
        X[:, i, 1] = i
        X[i, :, 2] = photos[0][1][i,:]
        X[i, :, 3] = photos[0][0][i,:]
    tools.show_image(X[1:,1:,[0,1,3]])
    cX = X
    for i in range(64):
        for j in range(64):
            cX[i, j, :3] = rotate.rotate_x(cX[i,j,0], cX[i,j,1], cX[i,j,2], 0.5)
    tools.show_image((cX-cX.min())[:,:,[0,1,3]])
from common import tools
from common.tools import IMG_SIZE
from common.db_helper import DBHelper, Database
from face_rotation import rotate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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


    helper = DBHelper()
    TOTAL_SUBJECTS_COUNT = helper.all_subjects_count()
    photos = []
    for database in helper.get_databases():
        if database.get_name() == 'ias_lab_rgbd':
            photos += load_samples(database, limit=2)

    X = np.zeros((IMG_SIZE, IMG_SIZE, 4))
    for i in range(64):
        X[i, :, 0] = i
        X[:, i, 1] = i
        X[i, :, 2] = photos[0][1][i,:]
        X[i, :, 3] = photos[0][0][i,:]

    cX = X
    for i in range(64):
        for j in range(64):
            if cX[i,j,2] > 0.8 or cX[i,j,2] < 0.01:
                cX[i,j,2] = np.median(cX[max(0, i-20):min(63, i+20),max(0, j-20):min(63, j+20),2])
                # TODO: take average of only closest pixels
    cX[:, :, 0] /= 64
    cX[:, :, 1] /= 64
    cX[:, :, 2] -= cX[:, :, 2].min()
    cX[:, :, 2] /= cX[:, :, 2].max()
    a3d = Axes3D(plt.figure())
    a3d.plot_surface(cX[:, :, 0], cX[:, :, 1], cX[:, :, 2], cmap=cm.coolwarm,)
    plt.show(a3d)
    for i in range(64):
        for j in range(64):
            cX[i, j, :3] = rotate.rotate_x(cX[i,j,0], cX[i,j,1], cX[i,j,2], 0.70)
    cX = (cX-cX.min())
    cX /= cX.max()
    print(cX)
    a3d = Axes3D(plt.figure())
    a3d.plot_surface(cX[:, :, 0], cX[:, :, 1], cX[:, :, 2], cmap=cm.coolwarm,)
    plt.show(a3d)
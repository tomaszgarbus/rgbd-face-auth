from skimage.feature import hog
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk


def get_hog_of(image: np.ndarray):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(1, 1), visualise=True)
    hog_image = hog_image/np.max(hog_image)
    return hog_image


def get_entropy_map_of(image: np.ndarray):
    entr = entropy(image, disk(3))
    entr = entr/np.max(entr)
    return entr

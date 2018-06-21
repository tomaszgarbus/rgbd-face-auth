# THIS FILE... needs more work.

from sympy import *
from PIL import Image
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import itertools
from common.tools import pic_with_applied_mask

import math

WIDTH = 960 # 1280
HEIGHT = 960
CROP_SIZE = (320, 0, 1280, 960)

def normalize(x, ntype):
    if ntype == 0:
        return [ e/255. for e in x ]
    if ntype == 1:
        n = math.sqrt(sum([ e**2 for e in x ])/len(x))
        return [ e/n for e in x ]

def ml_preproc(xs):
    ret = xs
    #ret = [ list(x) + [(i//WIDTH)//100, (i%WIDTH)//100] for i, x in enumerate(ret) ]
    #ret = [[x[0]-x[3], x[0]-x[6], x[3]-x[6], (x[0]+x[3])//2, (x[0]+x[6])//2, (x[3]+x[6])//2] for x in ret]
    ret = [ list(x) + [x[0]-x[3], x[0]-x[6], x[3]-x[6], (x[0]+x[3])//2, (x[0]+x[6])//2, (x[3]+x[6])//2] for x in ret]
    #ret = [ [x[0]/max(0.2, x[3]), x[3]/max(0.2, x[6]), x[0]/max(0.2, x[6])] for x in ret ]
    #ret = [ normalize(x, 0) for x in ret]

    return ret

def get_model(name_list):
    from sklearn import svm
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

    x_l, y_l = [], []
    n = 1

    for name in name_list:
        x = load_object(name)
        y = load_mask(name)
        x = ml_preproc( list(x.reshape(HEIGHT * WIDTH, 9)) )
        y = list(y.reshape(HEIGHT * WIDTH))
        x_l += list(x)[::n]
        y_l += list(y)[::n]
        #TO THINK: learn model here? Image by image.

    #clf = svm.SVC(kernel='poly', degree=1, verbose=True)
    clf = RandomForestClassifier(max_depth=27, n_jobs=7, verbose=1)
    #clf = GradientBoostingClassifier(verbose=True)
    #clf = AdaBoostClassifier()
    clf.fit(x_l, y_l)

    return clf

wave_d = [940, 890, 850]

def my_concatenate(a, b, c): #TODO
    return np.array([a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]])

def load_object(filename_base, extention = "jpg"):
    pictures = []
    for p_id in range(1, 3+1):
        with Image.open('./skin/' + filename_base + "_" + str(p_id) + '.' + extention) as im_frame:
            im_frame = im_frame.crop(CROP_SIZE)
            np_frame = np.array(im_frame.getdata())
            pictures += [np_frame.reshape(HEIGHT, WIDTH, 3)]
            # plt.imshow(pictures[-1].astype('byte') / 255.0)
            # plt.show()

    ret = np.zeros(shape=(HEIGHT, WIDTH, 9))
    for i, j in itertools.product(range(HEIGHT), range(WIDTH)):
        ret[i][j] = my_concatenate(pictures[0][i][j], pictures[1][i][j], pictures[2][i][j])

    return ret

def load_mask(filename_base, extention = "png"):
    with Image.open('./skin/' + filename_base + "_mask" + '.' + extention) as im_frame:
        im_frame2 = im_frame
        if filename_base == 'A': #TODO Check if RGBA or RGB
            im_frame2 = Image.new("RGB", im_frame.size, (255, 255, 255))
            im_frame2.paste(im_frame, mask=im_frame.split()[3])
        im_frame2 = im_frame2.crop(CROP_SIZE)
        np_frame = np.array(im_frame2.getdata())
        pic = np_frame.reshape(HEIGHT, WIDTH, 3)

    mask = np.zeros(shape=(HEIGHT, WIDTH), dtype=bool)
    for i, j in itertools.product(range(HEIGHT), range(WIDTH)):
        mask[i][j] = not is_black(*pic[i][j])

    return mask

def waves_to_rgb(pic):
    ret = np.zeros(shape=(HEIGHT, WIDTH, 3), dtype=int)

    for i, j in itertools.product(range(HEIGHT), range(WIDTH)):
        for k in range(3):
            ret[i][j][k] = sum([pic[i][j][k+3*p] for p in range(3)])//3

    return ret

def is_black(r, g, b):
    return r==0 or g == 0 or b == 0 or r**2 + g**2 + b**2 < 10**2

def is_vawe_balck(pixel):
    for i in range(3):
        if is_black(*pixel[3*i:3*(i+1)]):
            return True

    return False



m = get_model(['A', 'B'])

for name in ['A', 'H', 'I', 'B', 'C', 'D', 'E', 'F', 'G']:
    pic = load_object(name)
    pic = pic.reshape(HEIGHT*WIDTH, 9)

    mask = m.predict(ml_preproc(pic))
    pic = pic.reshape(HEIGHT, WIDTH, 9)
    mask = mask.reshape(HEIGHT, WIDTH)

    plt.imshow(pic_with_applied_mask(waves_to_rgb(pic), mask))
    plt.show()

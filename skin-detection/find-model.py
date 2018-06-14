# THIS FILE... needs more work.

from sympy import * 
from PIL import Image 
from random import randint 
import numpy as np 
import matplotlib.pyplot as plt
import itertools
from common.tools import pic_with_applied_mask

from sklearn import svm

def get_model(name_list):
    assert len(name_list) == 1, "tmpbad"
    x, y = None, None

    for name in name_list:
        x = load_object(name)
        y = load_mask(name)

    x = list(x.reshape(len(name_list) * 960 * 1280, 9))
    y = list(y.reshape(len(name_list) * 960 * 1280))

    print(len(x[0]), len(y))

    clf = svm.SVC(verbose=True)
    clf.fit(x[::100], y[::100])

    return clf

wave_d = [940, 890, 850]

def my_concatenate(a, b, c): #TODO
    return np.array([a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]])

def load_object(filename_base, extention = "jpg"):
    pictures = []
    for p_id in range(1, 3+1):
        with Image.open('./skin/' + filename_base + "_" + str(p_id) + '.' + extention) as im_frame: 
            np_frame = np.array(im_frame.getdata()) 
            pictures += [np_frame.reshape(960, 1280, 3)]
            #plt.imshow(pictures[-1])
            #plt.show()

    ret = np.zeros(shape=(960, 1280, 9))
    for i, j in itertools.product(range(960), range(1280)):
        ret[i][j] = my_concatenate(pictures[0][i][j], pictures[1][i][j], pictures[2][i][j])

    return ret

def load_mask(filename_base, extention = "png"):
    with Image.open('./skin/' + filename_base + "_mask" + '.' + extention) as im_frame:
        im_frame2 = Image.new("RGB", im_frame.size, (255, 255, 255))
        im_frame2.paste(im_frame, mask=im_frame.split()[3])
        np_frame = np.array(im_frame2.getdata())
        pic = np_frame.reshape(960, 1280, 3)

    mask = np.zeros(shape=(960, 1280), dtype=bool)
    for i, j in itertools.product(range(960), range(1280)):
        mask[i][j] = not is_black(*pic[i][j])

    return mask

def waves_to_rgb(pic):
    ret = np.zeros(shape=(960, 1280, 3), dtype=int)

    for i, j in itertools.product(range(960), range(1280)):
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



m = get_model(['A'])

pic = load_object('A')
pic = pic.reshape(960*1280, 9)

mask = m.predict(pic)
pic = pic.reshape(960, 1280, 9)
mask = mask.reshape(960, 1280)

plt.imshow(pic_with_applied_mask(waves_to_rgb(pic), mask))
plt.show()

exit(0)


pic = load_object('A')
pic_mask = load_mask('A')

#plt.imshow(pic_mask)
#plt.show()
#plt.imshow(pic_with_applied_mask(waves_to_rgb(pic), pic_mask))
#plt.show()

def preprocess(pixel): # 9 values wave-pixel
    tmp = []
    for i in range(3):
        a = pixel[0+i] / max(1, pixel[3+i])
        b= pixel[3+i]  / max(1, pixel[6+i])
        c = pixel[0+i] / max(1, pixel[6+i])


        # Potestować, jak liczyć pochodna
       
        #a = pixel[0][i] - pixel[1][i]
        #b = pixel[1][i] - pixel[2][i]
        #c = pixel[0][i] - pixel[2][i]

        #a /= wave_d[1] - wave_d[0]
        #b /= wave_d[2] - wave_d[1]
        #c /= wave_d[2] - wave_d[0]

        tmp += [a, b, c]


    return tmp

pictures = []
skin_pixels = []

for i, j in itertools.product(range(960), range(1280)):
    if pic_mask[i][j]:
        skin_pixels.append(pic[i][j])

derivative = [preprocess(x) for x in skin_pixels]

mid = [0] * 9
for x in derivative:
    for i in range(9):
        mid[i] += x[i]

for i in range(9):
    mid[i] /= len(derivative)

def check(d):
    mod = [(0.5, 1.5)] * 9
    for i in range(9):
        mi, ma = mod[i]
        if not (mi*mid[i] <= d[i] <= ma*mid[i]):
            return False

    return True

def generate_mask(pic):
    mask = np.zeros(shape=(960, 1280), dtype=bool)
    for i in range(960):
        for j in range(1280):
            mask[i][j] = check(preprocess(pic[i][j]))

    return mask

for test_name in ['A', 'B']:
    test_pic = load_object(test_name)
    mask = generate_mask(test_pic)
    pwam = pic_with_applied_mask(waves_to_rgb(test_pic), mask)
    plt.imshow(pwam)
    plt.show()

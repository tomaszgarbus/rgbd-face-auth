# THIS FILE... needs more work.

from sympy import * 
from PIL import Image 
from random import randint 
import numpy as np 
import matplotlib.pyplot as plt


wave_d = [940, 890, 850]

def is_black(x):
    return x[0]**2 + x[1]**2 + x[2]**2 < 10**2

pictures = []
skin_pixels = []

for filename in ['skin/C.jpg', 'skin/B.jpg', 'skin/A.jpg']:
    im_frame = Image.open('./' + filename) 
    np_frame = np.array(im_frame.getdata()) 
    pictures += [np_frame]

for i in range(len(pictures[0])):
    if not is_black(pictures[0][i]):
        skin_pixels += [ (pictures[0][i], pictures[1][i], pictures[2][i]) ]

derivative = []
for pixel in skin_pixels:
    tmp = []
    for i in range(3):
        a = pixel[0][i] / pixel[1][i]
        b = pixel[1][i] / pixel[2][i]
        c = pixel[0][i] / pixel[2][i]

        # Potestować, jak liczyć pochodna
       
        #a = pixel[0][i] - pixel[1][i]
        #b = pixel[1][i] - pixel[2][i]
        #c = pixel[0][i] - pixel[2][i]

        #a /= wave_d[1] - wave_d[0]
        #b /= wave_d[2] - wave_d[1]
        #c /= wave_d[2] - wave_d[0]

        tmp.append( [a, b, c] )

    for i in range(3): # to jest pewnie zle, ale na szybko tak...
        tmp[0][i] += tmp[2][i]
        tmp[0][i] /= 2

    tmp = tmp[:2]
    derivative.append(tmp)

def m_foo(a, b, foo):
    assert len(a) == len(b), "lab"
    return [ foo(a[i], b[i]) for i in range(len(a)) ]

def mdd(a, b):
    return a + b

m1, m2 = derivative[0][0], derivative[0][1]
k1, k2 = m1, m2
s1, s2 = [0, 0, 0], [0, 0, 0]
for d in derivative:
    m1 = m_foo(m1, d[0], min)
    m2 = m_foo(m2, d[1], min)
    k1 = m_foo(k1, d[0], max)
    k2 = m_foo(k2, d[1], max)
    s1 = m_foo(s1, d[0], mdd)
    s2 = m_foo(s2, d[0], mdd)

s1 = [x / len(derivative) for x in s1]
s2 = [x / len(derivative) for x in s2]

# Jeszcze tu bym wariancje policzyl

print(m1, m2)
print(s1, s2)
print(k1, k2)

def check(d):
    for i in range(len(s1)):
        if not (0.19 * s1[i] <= d[0][i] <= 1.92 * s1[i]):
            return False

    for i in range(len(s2)):
        if not (0.16* s2[i] <= d[1][i] <= 2.38 * s2[i]):
            return False

    return True

print(len([d for d in derivative if check(d)]) , "of", len(derivative))

abba = plt.imshow(pictures[0])

    pass

# THIS FILE... needs more work.

from sympy import * 
from PIL import Image 
from random import randint 
import numpy as np 
import matplotlib.pyplot as plt
import itertools
from common.tools import pic_with_applied_mask, rgb_skin_mark

import math

WIDTH = 450
HEIGHT = 550

def generate_mask_from_skin(pic) -> None:
    from itertools import product

    mask = np.zeros((HEIGHT, WIDTH), dtype=np.bool)
    mark = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float)

    for x, y in product(range(HEIGHT), range(WIDTH)):
        r, g, b = pic[x][y]
        mark[x][y] = rgb_skin_mark(r, g, b)

    probe = [mark[x][y] for x, y in product(range(4*HEIGHT//10, 5*HEIGHT//10, 8),
                                            range(4*WIDTH//10, 5*WIDTH//10, 8))]
    probe.sort(key=(lambda x: x[0]**2 + x[1]**2 + x[2]**2))
    A, B, C = probe[len(probe)//2]

    for x, y in product(range(HEIGHT), range(WIDTH)):
        a, b, c = mark[x][y]
        mi = 0.918
        ma = 1.092
        mask[x][y] = (0.3 * A <= a <= 4 * A) and (mi * B <= b <= ma * B) and (mi*C <= c <= ma*C)

    return mask

with Image.open('./skin/rgb-face.png') as im_frame:
    im_frame2 = Image.new("RGB", im_frame.size, (255, 255, 255))
    im_frame2.paste(im_frame, mask=im_frame.split()[3])
    np_frame = np.array(im_frame2.getdata()) 
    picture = np_frame.reshape(HEIGHT, WIDTH, 3)

mask = generate_mask_from_skin(picture)

nn = pic_with_applied_mask(picture, mask)
plt.imshow(nn)
plt.show()

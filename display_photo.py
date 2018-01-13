# Input file must be in a format described in data_format.md

import matplotlib.pyplot as plt
import numpy as np
import sys
from array import array

FORMATS = ['PHDE', 'PHIR']

filename = sys.argv[1]

with open(filename, 'rb') as f:
    format_arr = array('B', [])
    format_arr.fromfile(f, 4)
    assert ''.join(map(chr, format_arr)) in FORMATS

    size_arr = array('I', [])
    size_arr.fromfile(f, 2)
    width, height = size_arr

    data_arr = array('f', [])
    data_arr.fromfile(f, height * width)

photo = np.asarray(data_arr)
photo = photo.reshape(height, width)

plt.gray()
plt.imshow(photo)
plt.show()

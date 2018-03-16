import matplotlib.pyplot as plt
import numpy as np
import sys
from array import array

FORMATS = ['PHDE', 'PHIR']

filename = sys.argv[1]

with open(filename, 'rb') as f:
    format_arr = np.fromfile(f, dtype='i1', count=4)
    assert ''.join(map(chr, format_arr)) in FORMATS

    size_arr = np.fromfile(f, dtype='i4', count=2)
    width, height = size_arr
    print(size_arr)

    data_arr = np.fromfile(f, dtype='f', count=height * width)

photo = np.asarray(data_arr)
photo = photo.reshape(height, width)

plt.gray()
plt.imshow(photo)
plt.show()

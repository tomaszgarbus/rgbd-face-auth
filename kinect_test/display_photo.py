import matplotlib.pyplot as plt
import numpy as np
import sys

filename = sys.argv[1]

with open(filename, 'r') as f:
    content = f.read()
    height = content.count('\n')
    width = len(content.split()) // height

photo = np.fromfile(filename, sep=' ')
photo = photo.reshape(height, width)

plt.gray()
plt.imshow(photo)
plt.show()

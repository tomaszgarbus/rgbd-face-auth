from scipy import ndimage
import face_recognition
import PIL
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk

def load_color_image_from_file(filename):
    return np.array(PIL.Image.open(filename))

def change_image_mode(source_mode, output_mode, img):
    return np.array(PIL.Image.fromarray(img, mode=source_mode).convert(output_mode))

def rgb_image_resize(img, size):
    tmp = PIL.Image.fromarray(img, mode='RGB')
    tmp = tmp.resize(size)
    return np.array(tmp)

def gray_image_resize(img, size):
    tmp = PIL.Image.fromarray(img, mode='F')
    tmp = tmp.resize(size)
    return np.array(tmp)

def color_image_to_face(img):
    coords = face_recognition.face_locations(img)
    if len(coords) == 1:
        (x1,y1,x2,y2) = coords[0]
        return rgb_image_resize(img[x1:x2,y2:y1], img.shape[:2])
    else:
        return img# TODO: handle it

def show_image(img):
    plt.imshow(img)
    plt.show()

def show_gray_image(img):
    plt.gray()
    plt.imshow(img)
    plt.show()

def rgb_image_to_gray_entropy(img):
    img = change_image_mode('RGBA', 'L', img)
    img_shape = img.shape
    ret = entropy(img.reshape(img_shape), disk(10)).reshape(img_shape[0], img_shape[1], 1)
    return ret

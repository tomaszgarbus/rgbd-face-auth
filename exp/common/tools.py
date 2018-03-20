import face_recognition
import PIL
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from skimage.filters.rank import entropy
from skimage.morphology import disk

IMG_SIZE = 128


def rgb_skin_check(R, G, B):
    """
        Based on "Human Skin Detection by Visible and Near-Infrared Imaging"
        by Yusuke Kanzawa, Yoshikatsu Kimura, Takashi Naito

        MVA2011 IAPR Conference on Machine Vision Applications, June 13-15, 2011, Nara, JAPAN
    """
    """
        [ Y]   [ 16]   [ 0.257,  0.504,  0.098]   [R]
        [Cb] = [128] + [-0.148, -0.291,  0.439] x [G]
        [Cr]   [128]   [ 0.439, -0.368, -0.071]   [B]

        skin_pixel implies: Cb in [77; 127] and Cr in [133; 173]
    """

    _  =  16 +    0.257 * R +    0.504 * G +    0.098 * B
    Cb = 128 + (-0.148) * R + (-0.291) * G +    0.439 * B
    Cr = 128 +    0.439 * R + (-0.368) * G + (-0.071) * B

    return 77 <= Cb and Cb <= 127 and 133 <= Cr and Cr <= 173


def load_color_image_from_file(filename, skin_only=False):
    ret = np.array(PIL.Image.open(filename), dtype='uint8')

    if skin_only:
        for row in ret:
            for pixel in row:
                if not rgb_skin_check(pixel[0], pixel[1], pixel[2]):
                    pixel *= 0

    return ret

def load_depth_photo(path):
    with open(path, 'rb') as f:
        format_arr = np.fromfile(f, dtype='i1', count=4)
        assert ''.join(map(chr, format_arr)) == 'PHDE'
        size_arr = np.fromfile(f, dtype='i4', count=2)
        width, height = size_arr
        data_arr = np.fromfile(f, dtype='f', count=height * width)
        photo = np.asarray(data_arr)
        photo = photo.reshape(height, width)
        return photo


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
        return img  # TODO: handle it


def show_image(img):
    plt.imshow(img)
    plt.show()


def show_3d_plot(X):
    a3d = Axes3D(plt.figure())
    a3d.plot_surface(X[:, :, 0], X[:, :, 1], X[:, :, 2], cmap=cm.coolwarm, )
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

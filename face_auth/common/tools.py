import face_recognition
import PIL
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from skimage.filters.rank import entropy
from skimage.morphology import disk
import logging
from typing import List

from common.constants import SHOW_PLOTS


def rgb_skin_mark(r: float, g: float, b: float) -> (float, float, float):
    """
        Based on "Human Skin Detection by Visible and Near-Infrared Imaging"
        by Yusuke Kanzawa, Yoshikatsu Kimura, Takashi Naito

        MVA2011 IAPR Conference on Machine Vision Applications, June 13-15, 2011, Nara, JAPAN
    """
    """
        [ Y]   [ 16]   [ 0.257,  0.504,  0.098]   [R]
        [Cb] = [128] + [-0.148, -0.291,  0.439] x [G]
        [Cr]   [128]   [ 0.439, -0.368, -0.071]   [B]

        In paper skin_pixel implies: Cb in [77; 127] and Cr in [133; 173]
    """

    y  =  16 +    0.257 * r +    0.504 * g +    0.098 * b
    cb = 128 + (-0.148) * r + (-0.291) * g +    0.439 * b
    cr = 128 +    0.439 * r + (-0.368) * g + (-0.071) * b

    return (y, cb, cr)


def pic_with_applied_mask(pic: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ret = np.copy(pic)

    assert len(pic) == len(mask) and len(pic[0]) == len(mask[0]), "Mask must have same size as pic"

    for x in range(len(pic)):
        for y in range(len(pic[0])):
            if mask[x][y]:
                ret[x][y] = (255, int(ret[x][y][1]/1.4), int(ret[x][y][2]/1.4))

    return ret


def load_color_image_from_file(filename: str):
    return np.array(PIL.Image.open(filename), dtype='uint8')


def load_depth_photo(path: str) -> np.ndarray:
    """
        :param path:
        :return: 2D np.ndarray of floats
    """
    with open(path, 'rb') as f:
        format_arr = np.fromfile(f, dtype=np.int8, count=4)
        assert ''.join(map(chr, format_arr)) == 'PHDE'
        size_arr = np.fromfile(f, dtype=np.int32, count=2)
        width, height = size_arr
        data_arr = np.fromfile(f, dtype=np.float32, count=height * width)
        photo = np.asarray(data_arr)
        photo = photo.reshape(height, width)
        return photo


def load_ir_photo(path: str) -> np.ndarray:
    """
            :param path:
            :return: 2D np.ndarray of floats
        """
    with open(path, 'rb') as f:
        format_arr = np.fromfile(f, dtype=np.int8, count=4)
        assert ''.join(map(chr, format_arr)) == 'PHIR'
        size_arr = np.fromfile(f, dtype=np.int32, count=2)
        width, height = size_arr
        data_arr = np.fromfile(f, dtype=np.float32, count=height * width)
        photo = np.asarray(data_arr)
        photo = photo.reshape(height, width)
        return photo


def change_image_mode(source_mode: str, output_mode: str, img: np.ndarray) -> np.ndarray:
    return np.array(PIL.Image.fromarray(img, mode=source_mode).convert(output_mode))


def rgb_image_resize(img: np.ndarray, size: tuple((int, int))) -> np.ndarray:
    """
        :param img: 3D numpy array of shape (img_width, img_height, 3)
        :param size: (new_width, new_height)
        :return: resized image in RGB move
    """
    tmp = PIL.Image.fromarray(img, mode='RGB')
    tmp = tmp.resize(size)
    return np.array(tmp)


def gray_image_resize(img: np.ndarray, size: tuple) -> np.ndarray:
    tmp = PIL.Image.fromarray(img, mode='F')
    tmp = tmp.resize(size)
    return np.array(tmp)


def color_image_to_face(img: np.ndarray) -> np.ndarray:
    coords = face_recognition.face_locations(img)
    if len(coords) == 1:
        (x1,y1,x2,y2) = coords[0]
        return rgb_image_resize(img[x1:x2,y2:y1], img.shape[:2])
    else:
        return img  # TODO: handle it


def show_image(img: np.ndarray) -> None:
    if not SHOW_PLOTS:
        return
    plt.imshow(img)
    plt.show()


def show_position(image: np.ndarray, landmarks: dict, azimuth: tuple, face_center: tuple):
    if not SHOW_PLOTS:
        return
    img = np.copy(image)
    mxx = img.shape[0] - 1
    mxy = img.shape[1] - 1
    logging.debug(str(landmarks))
    for (key, v) in landmarks.items():
        (x, y, z) = v
        x *= (mxx + 1)
        y *= (mxy + 1)
        img[min(max(int(x), 0), mxx), min(max(int(y), 0), mxy)] = 1

    v = np.array([face_center[0]*(mxx + 1), face_center[1]*(mxy+1), face_center[2]])
    if azimuth is not None:
        azimuth = np.array([azimuth.item(0), azimuth.item(1), azimuth.item(2)])
        for i in range(100):
            x = min(max(int(v[0]), 0), mxx)
            y = min(max(int(v[1]), 0), mxy)
            img[x, y] = ((100-i)/100)
            if i % 10 == 0:
                logging.debug("point on " + str(x) + "," +str(y))
            v -= azimuth

    show_image(img)


def show_3d_plot(X: np.ndarray) -> None:
    """
        :param X: must be 3 dimensional numpy array, with last dimension
        of size >= 3
    """
    if not SHOW_PLOTS:
        return
    a3d = Axes3D(plt.figure())
    a3d.plot_surface(X[:, :, 0], X[:, :, 1], X[:, :, 2], cmap=cm.coolwarm, )
    plt.show()


def rgb_image_to_gray_entropy(img: np.ndarray) -> np.ndarray:
    img = change_image_mode('RGBA', 'L', img)
    img_shape = img.shape
    ret = entropy(img.reshape(img_shape), disk(10)).reshape(img_shape[0], img_shape[1], 1)
    return ret


def concat_images(images: List[np.ndarray]) -> np.ndarray:
    def _concat2(img1:np.ndarray, img2: np.ndarray, ax=0):
        if img1 is None:
            return img2
        elif img2 is None:
            return img1
        else:
            return np.concatenate((img1, img2), axis=ax)

    def _concat(img_list):
        con = None
        for img in img_list:
            con = _concat2(con, img)
        return con

    if len(images) % 2 == 0:
        con1 = _concat(images[::2])
        con2 = _concat(images[1::2])
        return _concat2(con1, con2, ax=1)

    return _concat(images)

# TODO: add saliency map to input vectors
# TODO(tomek): implement loading IR photos

# Unified class for loading images from databases
import tools
from tools import image_size, TYPES

import face_recognition
import matplotlib.pyplot as plt
import numpy as np
from array import array
import sys
import os
from skimage.filters.rank import entropy
from skimage.morphology import disk


DB_NAMES = ['www.vap.aau.dk', 'ias_lab_rgbd', 'superface_dataset']
SUBJECTS_COUNTS = {
    'www.vap.aau.dk': 31,
    'ias_lab_rgbd': 26,
    'superface_dataset': 18 # TODO(tomek): fix folder subject010 and change 18->20
}

def photo_to_face(color_photo, depth_photo):
    # Converts full photo to just face image
    # Locate face
    face_coords = face_recognition.face_locations(color_photo)
    # Process face detected by the library
    if len(face_coords) == 1:
        (x1,y1,x2,y2) = face_coords[0]
        # Cut out RGB & D face images
        depth_face = depth_photo[x1:x2,y2:y1]
        color_face = color_photo[x1:x2,y2:y1]
        # TODO: consider using landmarks to enrich input. Maybe not in this
        # function, but let's keep it in mind.
        landmarks = face_recognition.face_landmarks(color_photo)[0]
        depth_face = tools.gray_image_resize(depth_face, (image_size, image_size))
        depth_face = depth_face/np.max(depth_face)
        color_face = tools.rgb_image_resize(color_face, (image_size, image_size))
        # RGB -> grey
        grey_face = tools.change_image_mode('RGB', 'L', color_face)
        grey_face = grey_face/np.max(grey_face)

        return grey_face, depth_face
    else:
        # Face couldn't be detected
        return None, None

def vap_load_train_subject(subject_no, img_no):
    path_depth = 'www.vap.aau.dk/files/%d/0%02d_%d_d.depth' % (subject_no+1, 1+img_no/3, 1+img_no%3)
    path_color = 'www.vap.aau.dk/files/%d/0%02d_%d_c.png' % (subject_no+1, 1+img_no/3, 1+img_no%3)
    # Load depth and color photo
    color_photo = tools.load_color_image_from_file(path_color)
    depth_photo = tools.load_depth_photo(path_depth)
    # Resize to common size
    color_photo = tools.change_image_mode('RGBA', 'RGB', color_photo)
    color_photo = tools.rgb_image_resize(color_photo, (depth_photo.shape[1], depth_photo.shape[0]))
    return photo_to_face(color_photo, depth_photo)

def ias_load_train_subject(subject_no, img_no):
    path_depth = 'ias_lab_rgbd/files/training/0%02d_%02d.depth' % (subject_no, img_no)
    path_color = 'ias_lab_rgbd/files/training/0%02d_%02d.png' % (subject_no, img_no)
    # Load depth and color photo
    color_photo = tools.load_color_image_from_file(path_color)
    depth_photo = tools.load_depth_photo(path_depth)
    # Resize to common size
    color_photo = tools.rgb_image_resize(color_photo, (depth_photo.shape[1], depth_photo.shape[0]))
    return photo_to_face(color_photo, depth_photo)

def superface_load_train_subject(subject_no, img_no):
    # TODO(tomek): fix folder subject010 and remove this line
    if subject_no == 9:
        subject_no = 19
    path_color = 'superface_dataset/files/subject%03d/rgbFrame/frameRGB%d.png' % (subject_no+1, img_no+1)
    path_depth = 'superface_dataset/files/subject%03d/depthFrame/frameD%d.depth' % (subject_no+1, img_no+1)
    # Check if file exists. For some reason, this database lacks some files
    if not os.path.isfile(path_color) or not os.path.isfile(path_depth):
        return None, None

    # Load depth and color photo
    color_photo = tools.load_color_image_from_file(path_color)
    depth_photo = tools.load_depth_photo(path_depth)
    # Resize to common size
    color_photo = tools.rgb_image_resize(color_photo, (depth_photo.shape[1], depth_photo.shape[0]))
    return photo_to_face(color_photo, depth_photo)

class DBHelper:
    db_name = None
    subjects_count = None

    def __init__(self, db_name):
        assert db_name in DB_NAMES
        self.db_name = db_name
        self.subjects_count = SUBJECTS_COUNTS[db_name]

    def imgs_per_subject(self, subject_no):
        if self.db_name == 'www.vap.aau.dk':
            return 3 * 17
        elif self.db_name == 'ias_lab_rgbd':
            return 13
        elif self.db_name == 'superface_dataset':
            # TODO(tomek): fix folder subject010 and remove this line
            if subject_no == 9:
                subject_no = 19
            color_path = 'superface_dataset/files/subject%03d/rgbFrame' % (subject_no+1)
            depth_path = 'superface_dataset/files/subject%03d/depthFrame' % (subject_no+1)
            color_files = next(os.walk(color_path))[2]
            depth_files = next(os.walk(depth_path))[2]
            return max(len(color_files), len(depth_files))  # some files seem to be missing in depthFrame/* ...
        else:
            assert False, 'NOT IMPLEMENTED'

    def load_greyd_face(self, subject_no, img_no):
        # Returns grey and depth image for a subject, normalized to values [0;1]
        assert img_no in range(0, self.imgs_per_subject(subject_no)), 'img_no must be indexed from 0'
        if self.db_name == 'www.vap.aau.dk':
            return vap_load_train_subject(subject_no, img_no)
        if self.db_name == 'ias_lab_rgbd':
            return ias_load_train_subject(subject_no, img_no)
        if self.db_name == 'superface_dataset':
            return superface_load_train_subject(subject_no, img_no)

    def build_input_vector(self, subject_no, img_no):
        """ Concatenates: grey_face, depth_face, entr_grey_face, entr_depth_face"""
        (grey_face, depth_face) = self.load_greyd_face(subject_no, img_no)
        if grey_face is None or depth_face is None:
            return None
        tmp = np.zeros((TYPES * image_size, image_size))
        entr_grey_face = entropy(grey_face, disk(5))
        entr_grey_face = entr_grey_face/np.max(entr_grey_face)
        entr_depth_face = entropy(depth_face, disk(5))
        entr_depth_face = entr_depth_face/np.max(entr_depth_face)
        tmp[0:image_size] = depth_face
        tmp[image_size:image_size*2] = grey_face
        tmp[image_size*2:image_size*3] = entr_grey_face
        tmp[image_size*3:image_size*4] = entr_depth_face
        return tmp

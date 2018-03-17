# TODO(tomek): implement loading IR photos

# Unified class for loading images from databases
import common.tools as tools
from common.tools import IMG_SIZE

import face_recognition
import numpy as np
import os

DB_LOCATION = 'database'

def photo_to_greyd_face(color_photo, depth_photo):
    """ Converts full photo to just face image """
    # Resize to common size
    color_photo = tools.rgb_image_resize(color_photo, (depth_photo.shape[1], depth_photo.shape[0]))
    # Locate face
    face_coords = face_recognition.face_locations(color_photo)
    # Process face detected by the library
    if len(face_coords) == 1:
        (x1,y1,x2,y2) = face_coords[0]
        # Cut out RGB & D face images
        depth_face = depth_photo[x1:x2,y2:y1]
        color_face = color_photo[x1:x2,y2:y1]
        depth_face = tools.gray_image_resize(depth_face, (IMG_SIZE, IMG_SIZE))
        depth_face = depth_face/np.max(depth_face)
        color_face = tools.rgb_image_resize(color_face, (IMG_SIZE, IMG_SIZE))
        # RGB -> grey
        grey_face = tools.change_image_mode('RGB', 'L', color_face)
        grey_face = grey_face/np.max(grey_face)

        return grey_face, depth_face
    else:
        # Face couldn't be detected
        return None, None

class Database:
    """
    Class Database corresponds to a single valid database directory.
    """
    _name = None
    _load_png = True
    _load_depth = True
    _load_ir = False
    _subject_dirs = []  # Lists subdirectories of subjects in the database
    _imgs_of_subject = []  # Lists images of each subject in the database

    def __init__(self, name, load_png=True, load_depth=True, load_ir=False):
        """
        :param name: exact name of the directory of the database
        :param load_png: True if only images which come in .png format should be loaded
        :param load_depth: True if only images which come in .depth format should be loaded
        :param load_ir: True if only images which come in .ir format should be loaded

        Only the intersection over selected formats of images will be loaded.
        """
        assert not load_ir, "Loading IR photos not implemented"
        assert os.path.isdir('/'.join([DB_LOCATION, name, 'files'])), "No such database %s" % name
        self._name = name
        self._load_png = load_png
        self._load_depth = load_depth
        self._load_ir = load_ir

        # Initialize |_subject_dirs|
        path = '/'.join([DB_LOCATION, self._name, 'files'])
        self._subject_dirs = next(os.walk(path))[1]

        # List images of each subject
        def filenames_of_type(all_files, type):
            #  e.g. if all_files=['001_1.png', '001_2.ir'], type='png', returns ['001_1']
            filtered =  list(filter(lambda fname: fname.endswith('.'+type), all_files))
            filenames = list(map(lambda fname: os.path.splitext(fname)[0],  filtered))
            return filenames
        self._imgs_of_subject = [[] for i in range(self.subjects_count())]
        for (subject, i) in zip(self._subject_dirs, range(self.subjects_count())):
            path = '/'.join([DB_LOCATION, self._name, 'files', subject])
            all_files = next(os.walk(path))[2]
            sets = []
            if self._load_png:
                sets.append(set(filenames_of_type(all_files, 'png')))
            if self._load_depth:
                sets.append(set(filenames_of_type(all_files, 'depth')))
            if self._load_ir:
                sets.append(set(filenames_of_type(all_files, 'ir')))
            assert len(sets) > 0, "No format chosen?"
            # Intersect images found for each format
            self._imgs_of_subject[i] = sets[0]
            for s in sets[1:]:
                self._imgs_of_subject[i] = self._imgs_of_subject[i].intersection(s)
            self._imgs_of_subject[i] = list(self._imgs_of_subject[i])

        # Filter out only those subjects which have some images
        i = 0
        indices = []
        while i < len(self._subject_dirs):
            if self._imgs_of_subject[i] != []:
                indices.append(i)
            i += 1
        self._subject_dirs = list(np.array(self._subject_dirs)[indices])
        self._imgs_of_subject = list(np.array(self._imgs_of_subject)[indices])

    def get_name(self):
        return self._name

    def subjects_count(self):
        return len(self._subject_dirs)

    def imgs_per_subject(self, subject_no):
        assert 0 <= subject_no and subject_no < len(self._imgs_of_subject), "Invalid |subject_no|"
        return len(self._imgs_of_subject[subject_no])

    def load_subject(self, subject_no, img_no):
        path = '/'.join([DB_LOCATION, self._name, 'files', self._subject_dirs[subject_no], self._imgs_of_subject[subject_no][img_no]])
        path_color = path + '.png'
        path_depth = path + '.depth'
        path_ir = path + '.ir'
        loaded_imgs = []
        if self._load_png:
            assert os.path.isfile(path_color), "No such file %s " % path_color
            color_photo = tools.load_color_image_from_file(path_color)
            loaded_imgs.append(color_photo)
        if self._load_depth:
            assert os.path.isfile(path_color), "No such file %s " % path_depth
            depth_photo = tools.load_depth_photo(path_depth)
            loaded_imgs.append(depth_photo)
        # TODO: load IR photo too
        return tuple(loaded_imgs)

    def load_greyd_face(self, subject_no, img_no):
        """ Loads an image of a subject and cuts out their face with external lib. Only supports png+depth
        combination. """
        assert self._load_png and self._load_depth and not self._load_ir, "Only for RGB+D use"
        (color_photo, depth_photo) = self.load_subject(subject_no, img_no)
        return photo_to_greyd_face(color_photo, depth_photo)


class DBHelper:
    _databases = []

    def __init__(self, load_png=True, load_depth=True, load_ir=False):
        assert os.path.isdir(DB_LOCATION), "Please create directory (or symlink) %s" % DB_LOCATION
        db_names = next(os.walk(DB_LOCATION))[1]
        for db_name in db_names:
            if db_name == 'gen':
                continue
            self._databases.append(Database(db_name, load_png, load_depth, load_ir))

    def get_databases(self):
        return self._databases

    def all_subjects_count(self):
        ans = 0
        for db in self._databases:
            ans += db.subjects_count()
        return ans

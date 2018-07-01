# Face rotation
This directory contains code used for face preprocessing - trimming,
face angle detection, face rotation, dropping corner values and structure filters.

## `example.py`
A simple showcase, loading several faces, normalizing them and displaying (before
and after). Before running, set `SHOW_PLOTS = True` in `common/constants.py`.

## `find_angle.py`
Dedicated to face angle detection.

## `other.py`
Contains a helper code for constructing face points, which are useful for
detecting face angle and recentering.

## `recentre.py`
Recentering the face after rotation.

## `rotate.py`
Code for dropping corner values (extremely high or extremely low depth) and
rotation itself.

## `structure_filters.py`
HOGs and entropy maps, useful for reducing image distortions after rotation.

## `trim_face.py`
Finds a polygon around the face, and trims the face accordingly. Also sets
the field `Face.mask` - a map of pixels belonging to the face.
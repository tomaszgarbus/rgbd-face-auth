"""
    Constants to be used along all code.
"""

"""
    Size of the single image in both dimensions: width & height to be generally
    assumed in all code.
"""
IMG_SIZE = 128

"""
    Background color, after trimming the face.
"""
BGCOLOR = 0

"""
    Fraction of margin to be left when cutting out a square containing the face
    from the image.
"""
MARGIN_COEF = 0.1

"""
    Please link the latest version of the database under this relative location.
"""
DB_LOCATION = 'database'

"""
    test_suffixes.json is a file containing suffixes of filenames chosen
    for the test dataset. The format is as follows:
    {
        "database1": {
            "global": ["suffix1", "suffix2", "suffix3"]
            "overridden1": ["suffix4", "suffix5", "suffix6"]
        }
    }
    where:
    * "database1" is the name of directory containing the database
    * "suffix1" is a suffix of the filename without extension. For instance,
    if you wish to choose images 003_02.png and 003_02.depth for test set,
    "003_02" is the suffix you want to add to test_suffixes.json.
    * "overridden1" is the name of subdirectory for which you wish to override the
    set of suffixes.

    Same rules apply to frontal_photo_suffixes.json
"""
TEST_SUF_FNAME = 'test_suffixes.json'
FRONT_SUF_FNAME = 'frontal_photo_suffixes.json'

""" vector to which face surface should be orthogonal after rotation """
FACE_AZIMUTH = [0, 0, 1]

""" smoothing iterations after rotation """
SMOOTHEN_ITER = 15

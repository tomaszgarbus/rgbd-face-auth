from common.constants import NUM_CLASSES
from classifiers.hog_classifier import HogFaceClassifier
from experiments.templates.load_database import load_data

if __name__ == '__main__':
    # test hogs only
    classifier = HogFaceClassifier()
    X, Y = load_data()
    X = []
    Y = []
    classifier.test(X, Y)

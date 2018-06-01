from common.constants import NUM_CLASSES
from classifiers.hog_classifier import HogFaceClassifier
from experiments.templates.load_database import load_data
from experiments.hogs_only.constants import EXP_NAME, INPUT_SIZE
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    # test hogs only
    classifier = HogFaceClassifier()
    x_train, y_train, x_test, y_test = load_data(EXP_NAME, INPUT_SIZE)
    classifier.fit(x_train, y_train)
    ys = classifier.prediction(x_test)
    score = accuracy_score(y_test, ys)
    print("score is " + score)



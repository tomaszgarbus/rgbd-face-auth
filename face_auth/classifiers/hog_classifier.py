import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

from model.face import Face
from classifiers.classification_results import ClassificationResults


def get_face_hog(face: Face) -> np.ndarray:
    np.concatenate((face.hog_gir_fd, face.hog_depth_image), axis=0)


def from_hot_one(ys):
    return [np.argmax(y) for y in ys]


class HogFaceClassifier:

    svc_pipeline = Pipeline([
        # ('preprocess', FunctionTransformer(get_face_hog)),
        ('classifier', SVC(
            C=10, kernel='poly',
            gamma=1, shrinking=False, class_weight='balanced',
            probability=True, tol=0.001, cache_size=10000,
            max_iter=-1, verbose=0))
    ])

    et_pipeline = Pipeline([
        # ('preprocess', FunctionTransformer(get_face_hog)),
        ('classifier', ExtraTreesClassifier(
            n_estimators=10000,
            criterion='entropy',
            max_features=0.2, verbose=0, n_jobs=2))
    ])

    ens = VotingClassifier(estimators=[
        ('svc', svc_pipeline),
        ('et', et_pipeline),
    ], voting='soft', weights=[10, 1], n_jobs=2)

    def __init__(self, binary_classification: bool = False, params: dict = None):
        self.binary_classification = binary_classification
        #self.ens = self.et_pipeline
        print(str(self.ens.get_params().keys()))
        if params is not None:
            self.ens.set_params(**params)

    def fit(self, x, y):
        self.ens.fit(x, y)

    def save(self, file):
        joblib.dump(self.ens, file)

    def load(self, file):
        self.ens = joblib.load(file)

    def cv_test(self, x, y):
        score = cross_val_score(self.ens, x, y, cv=3, verbose=3, n_jobs=3)
        return score.mean()

    def prediction(self, X):
        return self.ens.predict(X)

    def evaluate(self, x_test, y_test) -> ClassificationResults:
        preds = self.prediction(x_test)
        pred_probs = self.ens.predict_proba(x_test)
        acc = accuracy_score(from_hot_one(y_test), preds)
        results = ClassificationResults(labels=y_test, preds=preds, pred_probs=pred_probs, acc=acc,
                                        binary=self.binary_classification)
        return results


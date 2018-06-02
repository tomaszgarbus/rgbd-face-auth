import numpy as np
from model.face import Face

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import FunctionTransformer



def getFaceHog(X: Face) -> np.ndarray:
    np.concatenate((X.hog_grey_fd, X.hog_depth_image), axis=0)


class HogFaceClassifier:

    svc_pipeline = Pipeline([
        #('preprocess', FunctionTransformer(getFaceHog)),
        ('classifier', SVC(
            C=1, kernel='rbf',
            gamma=1, shrinking=False,
            probability=True, tol=0.001, cache_size=1000,
            max_iter=-1, verbose=0))
    ])

    et_pipeline = Pipeline([
        #('preprocess', FunctionTransformer(getFaceHog)),
        ('classifier', ExtraTreesClassifier(
            n_estimators=5000,
            criterion='gini',
            max_features='auto', verbose=0))
    ])

    ens = VotingClassifier(estimators=[
        ('svc', svc_pipeline),
        ('et', et_pipeline),
    ], voting='soft', weights=[1, 10])

    def __init__(self):
        pass

    def fit(self, x, y):
        self.ens.fit(x, y)

    def save(self, file):
        joblib.dump(self.ens, file)

    def load(self, file):
        self.ens = joblib.load(file)

    def test(self, x, y):
        score = cross_val_score(self.ens, x, y, cv=3, verbose=3, n_jobs=3)
        return score.mean()

    def prediction(self, X):
        return self.ens.predict(X)

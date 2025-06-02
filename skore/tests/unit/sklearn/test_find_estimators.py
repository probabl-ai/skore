import numpy as np
import skrub
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion, make_pipeline
from skore import find_estimators


def test_find_estimators():
    estimators = [("linear_pca", PCA()), ("kernel_pca", KernelPCA())]
    combined = FeatureUnion(estimators)
    combined = FeatureUnion([("test", FeatureUnion(estimators)), ("one", PCA())])
    pip = make_pipeline(combined, RandomForestClassifier())

    extracted = find_estimators(pip)
    assert len(extracted) == 2
    assert extracted.get("ensemble") == ["RandomForestClassifier"]


def test_find_estimators_with_skrub():
    pip = skrub.tabular_learner("classification")
    extracted = find_estimators(pip)
    assert len(extracted) == 5
    assert extracted.get("_datetime_encoder") == ["DatetimeEncoder"]


def test_find_estimators_homemade():
    class MockRegressor(RegressorMixin, BaseEstimator):
        def __init__(self, n_call_predict=0):
            self.n_call_predict = n_call_predict

        def fit(self, X, y):
            self.fitted_ = True
            return self

        def predict(self, X):
            return np.ones(X.shape[0])

    pip = make_pipeline(MockRegressor())
    extracted = find_estimators(pip)
    assert len(extracted) == 1
    assert extracted.get("other") == ["MockRegressor"]

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_classification, make_regression


@pytest.fixture
def regression_data():
    return make_regression(n_features=5, random_state=42)


@pytest.fixture
def positive_regression_data():
    X, y = make_regression(n_features=5, random_state=42)
    return X, np.abs(y) + 0.1


@pytest.fixture
def multi_regression_data():
    return make_regression(n_features=5, n_targets=2, random_state=42)


@pytest.fixture
def classification_data():
    return make_classification(n_features=5, random_state=42)


@pytest.fixture
def outlier_data():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (300, 2))
    X[-10:] = rng.uniform(-10, 10, (10, 2))  # outliers
    return X, np.zeros(len(X))  # y as dummy zeros


@pytest.fixture
def clustering_data():
    return make_blobs(centers=3, n_features=2, random_state=42)

import pytest
from sklearn.datasets import make_classification, make_regression


@pytest.fixture
def regression_data():
    return make_regression(n_features=5, random_state=42)


@pytest.fixture
def classification_data():
    return make_classification(n_features=5, random_state=42)

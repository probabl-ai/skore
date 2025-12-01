import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import _convert_container

from skore._sklearn.feature_names import _get_feature_names


class Transformer:
    def fit(self, X, y):
        self.is_fitted = True
        return self

    def transform(self, X):
        return X


class Predictor:
    def fit(self, X, y):
        self.is_fitted = True
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])


def test_get_feature_names_error():
    """Check that a proper error message is raised when neither X nor n_features is
    provided and that we cannot infer feature names from the estimator."""

    predictor = Predictor()
    err_msg = "Feature names cannot be inferred from the estimator or transformer."
    with pytest.raises(ValueError, match=err_msg):
        _get_feature_names(predictor)


@pytest.mark.parametrize("container_type", ["array", "dataframe"])
def test_get_feature_names_minimal_model(container_type):
    """Test feature names inference for minimal models with numpy and pandas inputs."""
    X, y = np.random.randn(10, 10), np.random.randint(0, 2, size=10)
    columns_names = [f"Feature #{i}" for i in range(X.shape[1])]
    X = _convert_container(X, container_type, columns_name=columns_names)

    y = np.random.randint(0, 2, size=10)
    minimal_model = make_pipeline(Transformer(), Predictor()).fit(X, y)

    err_msg = "Feature names cannot be inferred from the estimator or transformer."
    with pytest.raises(ValueError, match=err_msg):
        _get_feature_names(estimator=minimal_model[-1], transformer=minimal_model[0])

    feature_names = _get_feature_names(
        estimator=minimal_model[-1], transformer=minimal_model[0], X=X
    )
    assert feature_names == columns_names

    feature_names = _get_feature_names(
        estimator=minimal_model[-1], transformer=minimal_model[:-1], X=X
    )
    assert feature_names == columns_names

    feature_names = _get_feature_names(
        estimator=minimal_model[-1], transformer=minimal_model[0], n_features=X.shape[1]
    )
    assert feature_names == columns_names


@pytest.mark.parametrize("container_type", ["array", "dataframe"])
def test_get_feature_names_sklearn_model(container_type):
    n_samples = 100
    X, y = np.random.randn(n_samples, 10), np.random.randint(0, 2, size=n_samples)
    columns_names = [f"Feature #{i}" for i in range(X.shape[1])]
    X = _convert_container(X, container_type, columns_name=columns_names)
    sklearn_model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)

    if container_type == "array":
        expected_feature_names = [f"x{i}" for i in range(X.shape[1])]
    else:
        expected_feature_names = columns_names

    feature_names = _get_feature_names(
        estimator=sklearn_model[-1], transformer=sklearn_model[0]
    )
    assert feature_names == expected_feature_names

    feature_names = _get_feature_names(
        estimator=sklearn_model[-1], transformer=sklearn_model[0], X=X
    )
    assert feature_names == expected_feature_names

    feature_names = _get_feature_names(
        estimator=sklearn_model[-1], transformer=sklearn_model[:-1], X=X
    )
    assert feature_names == expected_feature_names

    feature_names = _get_feature_names(
        estimator=sklearn_model[-1], transformer=sklearn_model[0], n_features=X.shape[1]
    )
    assert feature_names == expected_feature_names

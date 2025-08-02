import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@pytest.fixture
def binary_classification_data():
    return make_classification(random_state=42)


@pytest.fixture
def binary_classification_train_test_split(binary_classification_data):
    X, y = binary_classification_data
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def multiclass_classification_data():
    return make_classification(
        n_classes=3, n_clusters_per_class=1, random_state=42, n_informative=10
    )


@pytest.fixture
def multiclass_classification_train_test_split(multiclass_classification_data):
    X, y = multiclass_classification_data
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def regression_data():
    return make_regression(n_features=5, random_state=42)


@pytest.fixture
def positive_regression_data():
    X, y = make_regression(n_features=5, random_state=42)
    return X, np.abs(y) + 0.1


@pytest.fixture
def regression_multioutput_data():
    return make_regression(n_targets=2, n_features=5, random_state=42)


@pytest.fixture
def regression_train_test_split(regression_data):
    X, y = regression_data
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def positive_regression_train_test_split(positive_regression_data):
    X, y = positive_regression_data
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def regression_multioutput_train_test_split(regression_multioutput_data):
    X, y = regression_multioutput_data
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def forest_binary_classification_with_test(binary_classification_train_test_split):
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    return RandomForestClassifier().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def svc_binary_classification_with_test(binary_classification_train_test_split):
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    return SVC().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def forest_multiclass_classification_with_test(
    multiclass_classification_train_test_split,
):
    X_train, X_test, y_train, y_test = multiclass_classification_train_test_split
    return RandomForestClassifier().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def svc_multiclass_classification_with_test(multiclass_classification_train_test_split):
    X_train, X_test, y_train, y_test = multiclass_classification_train_test_split
    return SVC().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def pipeline_binary_classification_with_test(binary_classification_train_test_split):
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    estimator = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    return estimator.fit(X_train, y_train), X_test, y_test


@pytest.fixture
def linear_regression_with_test(regression_train_test_split):
    X_train, X_test, y_train, y_test = regression_train_test_split
    return LinearRegression().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def linear_regression_multioutput_with_test(regression_multioutput_train_test_split):
    X_train, X_test, y_train, y_test = regression_multioutput_train_test_split
    return LinearRegression().fit(X_train, y_train), X_test, y_test

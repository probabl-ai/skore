import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@pytest.fixture
def binary_classification_data():
    X, y = make_classification(class_sep=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return LogisticRegression().fit(X_train, y_train), X_train, X_test, y_train, y_test


@pytest.fixture
def multiclass_classification_data():
    X, y = make_classification(
        class_sep=0.1, n_classes=3, n_clusters_per_class=1, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return LogisticRegression().fit(X_train, y_train), X_train, X_test, y_train, y_test


@pytest.fixture
def binary_classification_data_no_split():
    X, y = make_classification(random_state=42)
    return LogisticRegression(), X, y


@pytest.fixture
def multiclass_classification_data_no_split():
    X, y = make_classification(n_classes=3, n_clusters_per_class=1, random_state=42)
    return LogisticRegression(), X, y

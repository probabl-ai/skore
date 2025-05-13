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


def get_roc_auc(
    display,
    label=None,
    split_number=None,
    estimator_name=None,
) -> float:
    noop_filter = display.roc_auc["roc_auc"].map(lambda x: True)
    label_filter = (display.roc_auc["label"] == label) if label is not None else True
    split_number_filter = (
        (display.roc_auc["split_index"] == split_number)
        if split_number is not None
        else True
    )
    estimator_name_filter = (
        (display.roc_auc["estimator_name"] == estimator_name)
        if estimator_name is not None
        else True
    )
    return display.roc_auc[
        noop_filter & label_filter & split_number_filter & estimator_name_filter
    ]["roc_auc"].iloc[0]

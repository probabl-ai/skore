import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from skore import ComparisonReport, CrossValidationReport, EstimatorReport


@pytest.fixture(scope="module")
def binary_classification():
    return make_classification(random_state=0, n_features=4)


@pytest.fixture(scope="module")
def binary_classification_train_test_split(binary_classification):
    X, y = binary_classification
    return train_test_split(X, y, test_size=0.2, random_state=0)


@pytest.fixture(scope="module")
def estimator_report_binary_classification_0(binary_classification_train_test_split):
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    return EstimatorReport(
        DummyClassifier(strategy="uniform", random_state=0),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@pytest.fixture(scope="module")
def estimator_report_binary_classification_1(binary_classification_train_test_split):
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    return EstimatorReport(
        DummyClassifier(strategy="uniform", random_state=1),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@pytest.fixture(scope="module")
def cross_validation_report_binary_classification_0(
    binary_classification,
):
    X, y = binary_classification
    return CrossValidationReport(
        DummyClassifier(strategy="uniform", random_state=0), X=X, y=y, splitter=2
    )


@pytest.fixture(scope="module")
def cross_validation_report_binary_classification_1(
    binary_classification,
):
    X, y = binary_classification
    return CrossValidationReport(
        DummyClassifier(strategy="uniform", random_state=1), X=X, y=y, splitter=2
    )


@pytest.fixture(scope="module")
def comparison_estimator_reports_binary_classification(
    estimator_report_binary_classification_0,
    estimator_report_binary_classification_1,
):
    return ComparisonReport(
        [
            estimator_report_binary_classification_0,
            estimator_report_binary_classification_1,
        ]
    )


@pytest.fixture(scope="module")
def comparison_cross_validation_reports_binary_classification(
    cross_validation_report_binary_classification_0,
    cross_validation_report_binary_classification_1,
):
    return ComparisonReport(
        [
            cross_validation_report_binary_classification_0,
            cross_validation_report_binary_classification_1,
        ]
    )


@pytest.fixture(scope="module")
def multiclass_classification():
    return make_classification(
        n_classes=3,
        n_clusters_per_class=1,
        n_informative=3,
        n_redundant=0,
        n_features=4,
        random_state=0,
    )


@pytest.fixture(scope="module")
def multiclass_classification_train_test_split(multiclass_classification):
    X, y = multiclass_classification
    return train_test_split(X, y, test_size=0.2, random_state=0)


@pytest.fixture(scope="module")
def estimator_report_multiclass_classification_0(
    multiclass_classification_train_test_split,
):
    X_train, X_test, y_train, y_test = multiclass_classification_train_test_split
    return EstimatorReport(
        DummyClassifier(strategy="uniform", random_state=0),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@pytest.fixture(scope="module")
def estimator_report_multiclass_classification_1(
    multiclass_classification_train_test_split,
):
    X_train, X_test, y_train, y_test = multiclass_classification_train_test_split
    return EstimatorReport(
        DummyClassifier(strategy="uniform", random_state=1),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@pytest.fixture(scope="module")
def cross_validation_report_multiclass_classification_0(
    multiclass_classification,
):
    X, y = multiclass_classification
    return CrossValidationReport(
        DummyClassifier(strategy="uniform", random_state=0), X=X, y=y, splitter=2
    )


@pytest.fixture(scope="module")
def cross_validation_report_multiclass_classification_1(
    multiclass_classification,
):
    X, y = multiclass_classification
    return CrossValidationReport(
        DummyClassifier(strategy="uniform", random_state=1), X=X, y=y, splitter=2
    )


@pytest.fixture(scope="module")
def comparison_estimator_reports_multiclass_classification(
    estimator_report_multiclass_classification_0,
    estimator_report_multiclass_classification_1,
):
    return ComparisonReport(
        [
            estimator_report_multiclass_classification_0,
            estimator_report_multiclass_classification_1,
        ]
    )


@pytest.fixture(scope="module")
def comparison_cross_validation_reports_multiclass_classification(
    cross_validation_report_multiclass_classification_0,
    cross_validation_report_multiclass_classification_1,
):
    return ComparisonReport(
        [
            cross_validation_report_multiclass_classification_0,
            cross_validation_report_multiclass_classification_1,
        ]
    )

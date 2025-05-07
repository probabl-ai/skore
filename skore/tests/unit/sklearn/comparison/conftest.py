import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from skore import ComparisonReport, CrossValidationReport, EstimatorReport


@pytest.fixture
def classification_data():
    X, y = make_classification(class_sep=0.1, random_state=42)
    return X, y


@pytest.fixture
def estimator_reports(classification_data):
    X, y = classification_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    estimator_report_1 = EstimatorReport(
        DummyClassifier(strategy="uniform", random_state=0),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    estimator_report_2 = EstimatorReport(
        DummyClassifier(strategy="uniform", random_state=1),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    return estimator_report_1, estimator_report_2


@pytest.fixture
def report_estimator_reports(estimator_reports):
    estimator_report_1, estimator_report_2 = estimator_reports
    return ComparisonReport([estimator_report_1, estimator_report_2])


@pytest.fixture
def cv_reports(classification_data):
    X, y = classification_data
    cv_report_1 = CrossValidationReport(
        DummyClassifier(strategy="uniform", random_state=0), X, y
    )
    cv_report_2 = CrossValidationReport(
        DummyClassifier(strategy="uniform", random_state=1), X, y
    )
    return cv_report_1, cv_report_2


@pytest.fixture
def report_cv_reports(cv_reports):
    cv_report_1, cv_report_2 = cv_reports
    return ComparisonReport([cv_report_1, cv_report_2])

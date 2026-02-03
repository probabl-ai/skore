import pytest
from sklearn.linear_model import LogisticRegression

from skore import ComparisonReport, EstimatorReport


@pytest.fixture
def comparison_report(binary_classification_train_test_split):
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    report_1 = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    return ComparisonReport(reports={"report_1": report_1, "report_2": report_2})


@pytest.fixture
def comparison_report_different_features(binary_classification_train_test_split):
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    report_1 = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    # Limit the number of features to the first 2
    report_2 = EstimatorReport(
        LogisticRegression(),
        X_train=X_train[:, 0:2],
        X_test=X_test[:, 0:2],
        y_train=y_train,
        y_test=y_test,
    )
    return ComparisonReport(reports={"report_1": report_1, "report_2": report_2})

import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from skore import ComparisonReport, CrossValidationReport, EstimatorReport


@pytest.fixture(scope="module")
def estimator_type():
    return "dummy"


def _classification_estimators(estimator_type, random_state_2=1):
    if estimator_type == "dummy":
        return (
            DummyClassifier(strategy="uniform", random_state=0),
            DummyClassifier(strategy="uniform", random_state=random_state_2),
        )
    if estimator_type == "linear":
        return (
            LogisticRegression(random_state=0),
            LogisticRegression(random_state=random_state_2),
        )
    if estimator_type == "forest":
        return (
            RandomForestClassifier(random_state=0),
            RandomForestClassifier(random_state=random_state_2),
        )
    raise ValueError(estimator_type)


def _regression_estimators(estimator_type, random_state_2=43):
    if estimator_type == "dummy":
        return DummyRegressor(), DummyRegressor()
    if estimator_type == "linear":
        return LinearRegression(), LinearRegression()
    if estimator_type == "forest":
        return (
            RandomForestRegressor(random_state=42),
            RandomForestRegressor(random_state=random_state_2),
        )
    raise ValueError(estimator_type)


@pytest.fixture(scope="module")
def binary_classification():
    return make_classification(random_state=0, n_features=4)


@pytest.fixture(scope="module")
def binary_classification_train_test_split(binary_classification):
    X, y = binary_classification
    return train_test_split(X, y, test_size=0.2, random_state=0)


@pytest.fixture(scope="module")
def estimator_reports_binary_classification(
    binary_classification_train_test_split, estimator_type
):
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    est_1, est_2 = _classification_estimators(estimator_type)
    return (
        EstimatorReport(
            est_1, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        ),
        EstimatorReport(
            est_2, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        ),
    )


@pytest.fixture(scope="module")
def cross_validation_reports_binary_classification(
    binary_classification, estimator_type
):
    X, y = binary_classification
    est_1, est_2 = _classification_estimators(estimator_type)
    return (
        CrossValidationReport(est_1, X=X, y=y, splitter=2),
        CrossValidationReport(est_2, X=X, y=y, splitter=2),
    )


@pytest.fixture(scope="module")
def comparison_estimator_reports_binary_classification(
    estimator_reports_binary_classification,
):
    report_1, report_2 = estimator_reports_binary_classification
    return ComparisonReport([report_1, report_2])


@pytest.fixture(scope="module")
def comparison_cross_validation_reports_binary_classification(
    cross_validation_reports_binary_classification,
):
    report_1, report_2 = cross_validation_reports_binary_classification
    return ComparisonReport([report_1, report_2])


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
def estimator_reports_multiclass_classification(
    multiclass_classification_train_test_split, estimator_type
):
    X_train, X_test, y_train, y_test = multiclass_classification_train_test_split
    est_1, est_2 = _classification_estimators(estimator_type)
    return (
        EstimatorReport(
            est_1, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        ),
        EstimatorReport(
            est_2, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        ),
    )


@pytest.fixture(scope="module")
def cross_validation_reports_multiclass_classification(
    multiclass_classification, estimator_type
):
    X, y = multiclass_classification
    est_1, est_2 = _classification_estimators(estimator_type)
    return (
        CrossValidationReport(est_1, X=X, y=y, splitter=2),
        CrossValidationReport(est_2, X=X, y=y, splitter=2),
    )


@pytest.fixture(scope="module")
def comparison_estimator_reports_multiclass_classification(
    estimator_reports_multiclass_classification,
):
    report_1, report_2 = estimator_reports_multiclass_classification
    return ComparisonReport([report_1, report_2])


@pytest.fixture(scope="module")
def comparison_cross_validation_reports_multiclass_classification(
    cross_validation_reports_multiclass_classification,
):
    report_1, report_2 = cross_validation_reports_multiclass_classification
    return ComparisonReport([report_1, report_2])


@pytest.fixture(scope="module")
def regression():
    return make_regression(n_features=4, random_state=0)


@pytest.fixture(scope="module")
def regression_train_test_split(regression):
    X, y = regression
    return train_test_split(X, y, test_size=0.2, random_state=0)


@pytest.fixture(scope="module")
def estimator_reports_regression(regression_train_test_split, estimator_type):
    X_train, X_test, y_train, y_test = regression_train_test_split
    est_1, est_2 = _regression_estimators(estimator_type)
    return (
        EstimatorReport(
            est_1, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        ),
        EstimatorReport(
            est_2, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        ),
    )


@pytest.fixture(scope="module")
def cross_validation_reports_regression(regression, estimator_type):
    X, y = regression
    est_1, est_2 = _regression_estimators(estimator_type)
    return (
        CrossValidationReport(est_1, X=X, y=y, splitter=2),
        CrossValidationReport(est_2, X=X, y=y, splitter=2),
    )


@pytest.fixture(scope="module")
def comparison_estimator_reports_regression(estimator_reports_regression):
    report_1, report_2 = estimator_reports_regression
    return ComparisonReport([report_1, report_2])


@pytest.fixture(scope="module")
def comparison_cross_validation_reports_regression(
    cross_validation_reports_regression,
):
    report_1, report_2 = cross_validation_reports_regression
    return ComparisonReport([report_1, report_2])


@pytest.fixture(scope="module")
def multioutput_regression():
    return make_regression(n_features=4, n_targets=2, random_state=0)


@pytest.fixture(scope="module")
def multioutput_regression_train_test_split(multioutput_regression):
    X, y = multioutput_regression
    return train_test_split(X, y, test_size=0.2, random_state=0)


@pytest.fixture(scope="module")
def estimator_reports_multioutput_regression(
    multioutput_regression_train_test_split, estimator_type
):
    X_train, X_test, y_train, y_test = multioutput_regression_train_test_split
    est_1, est_2 = _regression_estimators(estimator_type)
    return (
        EstimatorReport(
            est_1, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        ),
        EstimatorReport(
            est_2, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        ),
    )


@pytest.fixture(scope="module")
def cross_validation_reports_multioutput_regression(
    multioutput_regression, estimator_type
):
    X, y = multioutput_regression
    est_1, est_2 = _regression_estimators(estimator_type)
    return (
        CrossValidationReport(est_1, X=X, y=y, splitter=2),
        CrossValidationReport(est_2, X=X, y=y, splitter=2),
    )


@pytest.fixture(scope="module")
def comparison_estimator_reports_multioutput_regression(
    estimator_reports_multioutput_regression,
):
    report_1, report_2 = estimator_reports_multioutput_regression
    return ComparisonReport([report_1, report_2])


@pytest.fixture(scope="module")
def comparison_cross_validation_reports_multioutput_regression(
    cross_validation_reports_multioutput_regression,
):
    report_1, report_2 = cross_validation_reports_multioutput_regression
    return ComparisonReport([report_1, report_2])

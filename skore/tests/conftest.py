from datetime import datetime, timezone

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from skore import ComparisonReport, CrossValidationReport, EstimatorReport


def pytest_configure(config):
    # Use matplotlib agg backend during the tests including doctests
    import matplotlib

    matplotlib.use("agg")


@pytest.fixture(autouse=True)
def monkeypatch_tmpdir(monkeypatch, tmp_path):
    """
    Change ``TMPDIR`` used by ``tempfile.gettempdir()`` to point to ``tmp_path``, so
    that it is automatically deleted after use, with no impact on user's environment.

    Force the reload of the ``tempfile`` module to change the cached return of
    ``tempfile.gettempdir()``.

    https://docs.python.org/3/library/tempfile.html#tempfile.gettempdir
    """
    import importlib
    import tempfile

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    importlib.reload(tempfile)


@pytest.fixture
def mock_now():
    return datetime.now(tz=timezone.utc)


@pytest.fixture
def mock_nowstr(mock_now):
    return mock_now.isoformat()


@pytest.fixture
def MockDatetime(mock_now):
    class MockDatetime:
        def __init__(self, *args, **kwargs): ...

        @staticmethod
        def now(*args, **kwargs):
            return mock_now

    return MockDatetime


@pytest.fixture(scope="session")
def pyplot():
    """Setup and teardown fixture for matplotlib.

    This fixture closes the figures before and after running the functions.

    Returns
    -------
    pyplot : module
        The ``matplotlib.pyplot`` module.
    """
    from matplotlib import pyplot

    pyplot.close("all")
    yield pyplot
    pyplot.close("all")


@pytest.fixture
def binary_classification_data():
    return make_classification(random_state=42, n_features=4)


@pytest.fixture
def binary_classification_train_test_split(binary_classification_data):
    X, y = binary_classification_data
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def multiclass_classification_data():
    return make_classification(
        n_classes=3,
        n_clusters_per_class=1,
        n_informative=3,
        n_redundant=0,
        n_features=4,
        random_state=42,
    )


@pytest.fixture
def multiclass_classification_train_test_split(multiclass_classification_data):
    X, y = multiclass_classification_data
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def regression_data():
    return make_regression(n_features=4, random_state=42)


@pytest.fixture
def positive_regression_data():
    X, y = make_regression(n_features=4, random_state=42)
    return X, np.abs(y) + 0.1


@pytest.fixture
def regression_multioutput_data():
    return make_regression(n_targets=2, n_features=4, random_state=42)


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
def forest_binary_classification_data(binary_classification_data):
    X, y = binary_classification_data
    return RandomForestClassifier(), X, y


@pytest.fixture
def forest_binary_classification_with_test(binary_classification_train_test_split):
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    return RandomForestClassifier().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def forest_binary_classification_with_train_test(
    binary_classification_train_test_split,
):
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    return (
        RandomForestClassifier().fit(X_train, y_train),
        X_train,
        X_test,
        y_train,
        y_test,
    )


@pytest.fixture
def logistic_binary_classification_data(binary_classification_data):
    X, y = binary_classification_data
    return LogisticRegression(), X, y


@pytest.fixture
def logistic_binary_classification_with_test(binary_classification_train_test_split):
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    return LogisticRegression().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def logistic_binary_classification_with_train_test(
    binary_classification_train_test_split,
):
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    return (
        LogisticRegression().fit(X_train, y_train),
        X_train,
        X_test,
        y_train,
        y_test,
    )


@pytest.fixture
def svc_binary_classification_data(binary_classification_data):
    X, y = binary_classification_data
    return SVC(), X, y


@pytest.fixture
def svc_binary_classification_with_test(binary_classification_train_test_split):
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    return SVC().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def svc_binary_classification_with_train_test(binary_classification_train_test_split):
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    return SVC().fit(X_train, y_train), X_train, X_test, y_train, y_test


@pytest.fixture
def forest_multiclass_classification_data(multiclass_classification_data):
    X, y = multiclass_classification_data
    return RandomForestClassifier(), X, y


@pytest.fixture
def forest_multiclass_classification_with_test(
    multiclass_classification_train_test_split,
):
    X_train, X_test, y_train, y_test = multiclass_classification_train_test_split
    return RandomForestClassifier().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def forest_multiclass_classification_with_train_test(
    multiclass_classification_train_test_split,
):
    X_train, X_test, y_train, y_test = multiclass_classification_train_test_split
    return (
        RandomForestClassifier().fit(X_train, y_train),
        X_train,
        X_test,
        y_train,
        y_test,
    )


@pytest.fixture
def logistic_multiclass_classification_data(multiclass_classification_data):
    X, y = multiclass_classification_data
    return LogisticRegression(), X, y


@pytest.fixture
def logistic_multiclass_classification_with_test(
    multiclass_classification_train_test_split,
):
    X_train, X_test, y_train, y_test = multiclass_classification_train_test_split
    return LogisticRegression().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def logistic_multiclass_classification_with_train_test(
    multiclass_classification_train_test_split,
):
    X_train, X_test, y_train, y_test = multiclass_classification_train_test_split
    return (
        LogisticRegression().fit(X_train, y_train),
        X_train,
        X_test,
        y_train,
        y_test,
    )


@pytest.fixture
def svc_multiclass_classification_data(multiclass_classification_data):
    X, y = multiclass_classification_data
    return SVC(), X, y


@pytest.fixture
def svc_multiclass_classification_with_test(multiclass_classification_train_test_split):
    X_train, X_test, y_train, y_test = multiclass_classification_train_test_split
    return SVC().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def svc_multiclass_classification_with_train_test(
    multiclass_classification_train_test_split,
):
    X_train, X_test, y_train, y_test = multiclass_classification_train_test_split
    return SVC().fit(X_train, y_train), X_train, X_test, y_train, y_test


@pytest.fixture
def pipeline_binary_classification_data(binary_classification_data):
    X, y = binary_classification_data
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())]), X, y


@pytest.fixture
def pipeline_binary_classification_with_test(binary_classification_train_test_split):
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    estimator = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    return estimator.fit(X_train, y_train), X_test, y_test


@pytest.fixture
def pipeline_binary_classification_with_train_test(
    binary_classification_train_test_split,
):
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    estimator = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    return estimator.fit(X_train, y_train), X_train, X_test, y_train, y_test


@pytest.fixture
def linear_regression_data(regression_data):
    X, y = regression_data
    return LinearRegression(), X, y


@pytest.fixture
def linear_regression_with_test(regression_train_test_split):
    X_train, X_test, y_train, y_test = regression_train_test_split
    return LinearRegression().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def linear_regression_with_train_test(regression_train_test_split):
    X_train, X_test, y_train, y_test = regression_train_test_split
    return LinearRegression().fit(X_train, y_train), X_train, X_test, y_train, y_test


@pytest.fixture
def linear_regression_multioutput_data(regression_multioutput_data):
    X, y = regression_multioutput_data
    return LinearRegression(), X, y


@pytest.fixture
def linear_regression_multioutput_with_test(regression_multioutput_train_test_split):
    X_train, X_test, y_train, y_test = regression_multioutput_train_test_split
    return LinearRegression().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def linear_regression_multioutput_with_train_test(
    regression_multioutput_train_test_split,
):
    X_train, X_test, y_train, y_test = regression_multioutput_train_test_split
    return LinearRegression().fit(X_train, y_train), X_train, X_test, y_train, y_test


@pytest.fixture
def cross_validation_report_binary_classification(forest_binary_classification_data):
    estimator, X, y = forest_binary_classification_data
    return CrossValidationReport(estimator, X, y, splitter=3)


@pytest.fixture
def cross_validation_report_multiclass_classification(
    forest_multiclass_classification_data,
):
    estimator, X, y = forest_multiclass_classification_data
    return CrossValidationReport(estimator, X, y, splitter=3)


@pytest.fixture
def estimator_reports_binary_classification(binary_classification_train_test_split):
    X_train, X_test, y_train, y_test = binary_classification_train_test_split

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
def estimator_reports_multiclass_classification(
    multiclass_classification_train_test_split,
):
    X_train, X_test, y_train, y_test = multiclass_classification_train_test_split

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
def comparison_estimator_reports_binary_classification(
    estimator_reports_binary_classification,
):
    estimator_report_1, estimator_report_2 = estimator_reports_binary_classification
    return ComparisonReport([estimator_report_1, estimator_report_2])


@pytest.fixture
def comparison_estimator_reports_multiclass_classification(
    estimator_reports_multiclass_classification,
):
    estimator_report_1, estimator_report_2 = estimator_reports_multiclass_classification
    return ComparisonReport([estimator_report_1, estimator_report_2])


@pytest.fixture
def cross_validation_reports_binary_classification(binary_classification_data):
    X, y = binary_classification_data
    cv_report_1 = CrossValidationReport(
        DummyClassifier(strategy="uniform", random_state=0), X, y, splitter=2
    )
    cv_report_2 = CrossValidationReport(
        DummyClassifier(strategy="uniform", random_state=1), X, y, splitter=2
    )
    return cv_report_1, cv_report_2


@pytest.fixture
def cross_validation_reports_multiclass_classification(multiclass_classification_data):
    X, y = multiclass_classification_data
    cv_report_1 = CrossValidationReport(
        DummyClassifier(strategy="uniform", random_state=0), X, y, splitter=2
    )
    cv_report_2 = CrossValidationReport(
        DummyClassifier(strategy="uniform", random_state=1), X, y, splitter=2
    )
    return cv_report_1, cv_report_2


@pytest.fixture
def comparison_cross_validation_reports_binary_classification(
    cross_validation_reports_binary_classification,
):
    cv_report_1, cv_report_2 = cross_validation_reports_binary_classification
    return ComparisonReport([cv_report_1, cv_report_2])


@pytest.fixture
def comparison_cross_validation_reports_multiclass_classification(
    cross_validation_reports_multiclass_classification,
):
    cv_report_1, cv_report_2 = cross_validation_reports_multiclass_classification
    return ComparisonReport([cv_report_1, cv_report_2])


@pytest.fixture
def estimator_reports_regression(regression_train_test_split):
    X_train, X_test, y_train, y_test = regression_train_test_split

    estimator_report_1 = EstimatorReport(
        DummyRegressor(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    estimator_report_2 = EstimatorReport(
        DummyRegressor(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    return estimator_report_1, estimator_report_2


@pytest.fixture
def comparison_estimator_reports_regression(
    estimator_reports_regression,
):
    estimator_report_1, estimator_report_2 = estimator_reports_regression
    return ComparisonReport([estimator_report_1, estimator_report_2])


@pytest.fixture
def cross_validation_reports_regression(regression_data):
    X, y = regression_data
    cv_report_1 = CrossValidationReport(DummyRegressor(), X, y)
    cv_report_2 = CrossValidationReport(DummyRegressor(), X, y)
    return cv_report_1, cv_report_2


@pytest.fixture
def comparison_cross_validation_reports_regression(
    cross_validation_reports_regression,
):
    cv_report_1, cv_report_2 = cross_validation_reports_regression
    return ComparisonReport([cv_report_1, cv_report_2])


@pytest.fixture
def linear_regression_comparison_report(linear_regression_with_train_test):
    """Fixture providing a ComparisonReport with two linear regression estimators."""
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
    estimator_2 = clone(estimator).fit(X_train, y_train)
    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator_2,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        }
    )
    return report

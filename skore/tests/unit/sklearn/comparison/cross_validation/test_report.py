import copy
import re
from io import BytesIO

import joblib
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from skore import ComparisonReport, CrossValidationReport


@pytest.fixture
def classification_data():
    X, y = make_classification(class_sep=0.1, random_state=42)
    return X, y


@pytest.fixture
def cv_report_classification(classification_data):
    X, y = classification_data
    cv_report = CrossValidationReport(DummyClassifier(), X, y)
    return cv_report


@pytest.fixture
def cv_report_regression():
    X, y = make_regression(random_state=42)
    cv_report = CrossValidationReport(DummyRegressor(), X, y)
    return cv_report


def test_init_wrong_parameters(cv_report_classification):
    """If the input is not valid, raise."""

    with pytest.raises(TypeError, match="Expected reports to be an iterable"):
        ComparisonReport(cv_report_classification)

    with pytest.raises(ValueError, match="Expected at least 2 reports to compare"):
        ComparisonReport([cv_report_classification])

    with pytest.raises(
        TypeError,
        match="Expected instances of EstimatorReport or CrossValidationReport",
    ):
        ComparisonReport([None, cv_report_classification])


def test_init_different_ml_usecases(cv_report_classification, cv_report_regression):
    """Raise an error if the passed estimators do not have the same ML usecase."""
    with pytest.raises(
        ValueError, match="Expected all estimators to have the same ML usecase"
    ):
        ComparisonReport([cv_report_regression, cv_report_classification])


def test_init_non_distinct_reports(cv_report_classification):
    """If the passed estimators are not distinct objects, raise an error."""

    with pytest.raises(ValueError, match="Expected reports to be distinct objects"):
        ComparisonReport([cv_report_classification, cv_report_classification])


def test_non_string_report_names(cv_report_classification):
    """If the estimators are passed as a dict with non-string keys,
    then the estimator names are the dict keys converted to strings."""

    report = ComparisonReport(
        {0: cv_report_classification, "1": copy.copy(cv_report_classification)}
    )
    assert report.report_names_ == ["0", "1"]


@pytest.fixture
def report(cv_report_classification):
    return ComparisonReport(
        [cv_report_classification, copy.copy(cv_report_classification)]
    )


def test_help(capsys, report):
    """Check the help menu works."""
    report.help()

    captured = capsys.readouterr()
    assert "Tools to compare estimators" in captured.out

    # Check that we have a line with accuracy and the arrow associated with it
    assert re.search(
        r"\.accuracy\([^)]*\).*\(↗︎\).*-.*accuracy", captured.out, re.MULTILINE
    )


def test_repr(report):
    """Check the `__repr__` works."""

    assert "ComparisonReport" in repr(report)


def test_metrics_repr(report):
    """Check the repr method of `report.metrics`."""
    repr_str = repr(report.metrics)
    assert "skore.ComparisonReport.metrics" in repr_str
    assert "help()" in repr_str


def test_pickle(tmp_path, report):
    """Check that we can pickle a comparison report."""
    with BytesIO() as stream:
        joblib.dump(report, stream)


def test_metrics_help(capsys, report):
    """Check that the help method writes to the console."""
    report.metrics.help()
    captured = capsys.readouterr()
    assert "Available metrics methods" in captured.out


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_get_predictions(report, classification_data, data_source):
    predictions = report.get_predictions(
        data_source=data_source, response_method="predict"
    )

    assert len(predictions) == len(report.reports_)
    for i, cv_report in enumerate(report.reports_):
        assert len(predictions[i]) == cv_report._cv_splitter.n_splits


def test_get_predictions_X_y(report, classification_data):
    with pytest.raises(NotImplementedError):
        report.get_predictions(data_source="X_y", response_method="predict")

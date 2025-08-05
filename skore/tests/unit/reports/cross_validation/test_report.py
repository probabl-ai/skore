import inspect
import re

import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from skore import CrossValidationReport, EstimatorReport
from skore._sklearn._cross_validation.report import _generate_estimator_report


def test_report_can_be_rebuilt_using_parameters(
    cross_validation_reports_binary_classification,
):
    report, _ = cross_validation_reports_binary_classification
    parameters = {}

    assert isinstance(report, CrossValidationReport)

    for parameter in inspect.signature(CrossValidationReport).parameters:
        assert hasattr(report, parameter), f"The parameter '{parameter}' must be stored"

        parameters[parameter] = getattr(report, parameter)

    CrossValidationReport(**parameters)


def test_generate_estimator_report(forest_binary_classification_data):
    """Test the behaviour of `_generate_estimator_report`."""
    estimator, X, y = forest_binary_classification_data
    # clone the estimator to avoid a potential side effect even though we check that
    # the report is not altering the estimator
    estimator = clone(estimator)
    train_indices = np.arange(len(X) // 2)
    test_indices = np.arange(len(X) // 2, len(X))
    report = _generate_estimator_report(
        estimator=RandomForestClassifier(n_estimators=2, random_state=42),
        X=X,
        y=y,
        train_indices=train_indices,
        test_indices=test_indices,
        pos_label=1,
    )

    assert isinstance(report, EstimatorReport)
    assert report.estimator_ is not estimator
    assert isinstance(report.estimator_, RandomForestClassifier)
    try:
        check_is_fitted(report.estimator_)
    except NotFittedError as exc:
        raise AssertionError("The estimator in the report should be fitted.") from exc
    np.testing.assert_allclose(report.X_train, X[train_indices])
    np.testing.assert_allclose(report.y_train, y[train_indices])
    np.testing.assert_allclose(report.X_test, X[test_indices])
    np.testing.assert_allclose(report.y_test, y[test_indices])


@pytest.mark.parametrize("cv", [5, 10])
@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize(
    "fixture_name",
    ["forest_binary_classification_data", "pipeline_binary_classification_data"],
)
def test_cross_validation_report_attributes(fixture_name, request, cv, n_jobs):
    """Test the attributes of the cross-validation report."""
    estimator, X, y = request.getfixturevalue(fixture_name)
    report = CrossValidationReport(estimator, X, y, cv_splitter=cv, n_jobs=n_jobs)
    assert isinstance(report, CrossValidationReport)
    assert isinstance(report.estimator_reports_, list)
    for estimator_report in report.estimator_reports_:
        assert isinstance(estimator_report, EstimatorReport)
    assert report.X is X
    assert report.y is y
    assert report.n_jobs == n_jobs
    assert len(report.estimator_reports_) == cv
    if isinstance(estimator, Pipeline):
        assert report.estimator_name_ == estimator[-1].__class__.__name__
    else:
        assert report.estimator_name_ == estimator.__class__.__name__

    with pytest.raises(AttributeError):
        report.estimator_ = LinearRegression()
    with pytest.raises(AttributeError):
        report.X = X
    with pytest.raises(AttributeError):
        report.y = y


def test_cross_validation_report_help(capsys, forest_binary_classification_data):
    """Check that the help method writes to the console."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X, y)

    report.help()
    captured = capsys.readouterr()
    assert f"Tools to diagnose estimator {estimator.__class__.__name__}" in captured.out

    # Check that we have a line with accuracy and the arrow associated with it
    assert re.search(
        r"\.accuracy\([^)]*\).*\(↗︎\).*-.*accuracy", captured.out, re.MULTILINE
    )


def test_cross_validation_report_repr(forest_binary_classification_data):
    """Check that __repr__ returns a string starting with the expected prefix."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X, y)

    repr_str = repr(report)
    assert "CrossValidationReport" in repr_str


@pytest.mark.parametrize(
    "fixture_name, expected_n_keys",
    [
        ("forest_binary_classification_data", 10),
        ("svc_binary_classification_data", 10),
        ("forest_multiclass_classification_data", 12),
        ("linear_regression_data", 4),
    ],
)
@pytest.mark.parametrize("n_jobs", [None, 1, 2])
def test_cross_validation_report_cache_predictions(
    request, fixture_name, expected_n_keys, n_jobs
):
    """Check that calling cache_predictions fills the cache."""
    estimator, X, y = request.getfixturevalue(fixture_name)
    report = CrossValidationReport(estimator, X, y, cv_splitter=2, n_jobs=n_jobs)
    report.cache_predictions(n_jobs=n_jobs)
    # no effect on the actual cache of the cross-validation report but only on the
    # underlying estimator reports
    assert report._cache == {}

    for estimator_report in report.estimator_reports_:
        assert len(estimator_report._cache) == expected_n_keys

    report.clear_cache()
    assert report._cache == {}
    for estimator_report in report.estimator_reports_:
        assert estimator_report._cache == {}


@pytest.mark.parametrize("data_source", ["train", "test", "X_y"])
@pytest.mark.parametrize(
    "response_method", ["predict", "predict_proba", "decision_function"]
)
@pytest.mark.parametrize("pos_label", [None, 0, 1])
def test_cross_validation_report_get_predictions(
    data_source, response_method, pos_label, logistic_binary_classification_data
):
    """Check the behaviour of the `get_predictions` method."""
    estimator, X, y = logistic_binary_classification_data
    report = CrossValidationReport(estimator, X, y, cv_splitter=2)

    if data_source == "X_y":
        predictions = report.get_predictions(
            data_source=data_source,
            response_method=response_method,
            X=X,
            pos_label=pos_label,
        )
    else:
        predictions = report.get_predictions(
            data_source=data_source,
            response_method=response_method,
            pos_label=pos_label,
        )
    assert len(predictions) == 2
    for split_idx, split_predictions in enumerate(predictions):
        if data_source == "train":
            expected_shape = report.estimator_reports_[split_idx].y_train.shape
        elif data_source == "test":
            expected_shape = report.estimator_reports_[split_idx].y_test.shape
        else:  # data_source == "X_y"
            expected_shape = (X.shape[0],)
        assert split_predictions.shape == expected_shape


def test_cross_validation_report_get_predictions_error(
    logistic_binary_classification_data,
):
    """Check that we raise an error when the data source is invalid."""
    estimator, X, y = logistic_binary_classification_data
    report = CrossValidationReport(estimator, X, y, cv_splitter=2)

    with pytest.raises(ValueError, match="Invalid data source"):
        report.get_predictions(data_source="invalid")

    with pytest.raises(ValueError, match="The `X` parameter is required"):
        report.get_predictions(data_source="X_y")


def test_cross_validation_report_pickle(tmp_path, binary_classification_data):
    """Check that we can pickle an cross-validation report.

    In particular, the progress bar from rich are pickable, therefore we trigger
    the progress bar to be able to test that the progress bar is pickable.
    """
    estimator, X, y = binary_classification_data
    report = CrossValidationReport(estimator, X, y, cv_splitter=2)
    report.cache_predictions()
    joblib.dump(report, tmp_path / "report.joblib")

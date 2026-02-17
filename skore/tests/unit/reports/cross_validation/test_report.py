import inspect

import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils._testing import _convert_container
from sklearn.utils.validation import check_is_fitted

from skore import CrossValidationReport, EstimatorReport
from skore._sklearn._cross_validation.report import _generate_estimator_report
from skore._utils._testing import MockEstimator


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
def test_attributes(fixture_name, request, cv, n_jobs):
    """Test the attributes of the cross-validation report."""
    estimator, X, y = request.getfixturevalue(fixture_name)
    report = CrossValidationReport(estimator, X, y, splitter=cv, n_jobs=n_jobs)
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
def test_cache_predictions(request, fixture_name, expected_n_keys, n_jobs):
    """Check that calling cache_predictions fills the cache."""
    estimator, X, y = request.getfixturevalue(fixture_name)
    report = CrossValidationReport(estimator, X, y, splitter=2, n_jobs=n_jobs)
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
def test_get_predictions(
    data_source, response_method, pos_label, logistic_binary_classification_data
):
    """Check the behaviour of the `get_predictions` method."""
    estimator, X, y = logistic_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)

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


def test_get_predictions_error(
    logistic_binary_classification_data,
):
    """Check that we raise an error when the data source is invalid."""
    estimator, X, y = logistic_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)

    with pytest.raises(ValueError, match="Invalid data source"):
        report.get_predictions(data_source="invalid")

    with pytest.raises(ValueError, match="The `X` parameter is required"):
        report.get_predictions(data_source="X_y")


def test_pickle(tmp_path, logistic_binary_classification_data):
    """Check that we can pickle an cross-validation report.

    In particular, the progress bar from rich are pickable, therefore we trigger
    the progress bar to be able to test that the progress bar is pickable.
    """
    estimator, X, y = logistic_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)
    report.cache_predictions()
    joblib.dump(report, tmp_path / "report.joblib")


@pytest.mark.parametrize(
    "error",
    [
        ValueError("No more fitting"),
        KeyboardInterrupt(),
    ],
)
@pytest.mark.parametrize("n_jobs", [None, 1, 2])
def test_interrupted_propagates_error(binary_classification_data, error, n_jobs):
    """Check that when a split fails during cross-validation, the error propagates."""
    X, y = binary_classification_data

    estimator = MockEstimator(error=error, n_call=0, fail_after_n_clone=8)

    with pytest.raises(type(error), match=str(error) if str(error) else None):
        CrossValidationReport(estimator, X, y, splitter=10, n_jobs=n_jobs)


def test_clustering():
    """Check that we cannot create a report with a clustering model."""
    with pytest.raises(
        ValueError,
        match="skore does not support clustering models "
        "yet. Please use a classification or regression model instead.",
    ):
        CrossValidationReport(KMeans(), X=np.random.rand(10, 5))


@pytest.mark.parametrize(
    "container_types", [("dataframe", "series"), ("array", "array")]
)
def test_create_estimator_report(container_types, forest_binary_classification_data):
    """Test the `create_estimator_report` method."""
    estimator, X, y = forest_binary_classification_data
    X = _convert_container(X, container_types[0])
    y = _convert_container(y, container_types[1])
    X_experiment, X_heldout, y_experiment, y_heldout = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    cv_report = CrossValidationReport(estimator, X_experiment, y_experiment, splitter=2)
    est_report = cv_report.create_estimator_report()

    assert isinstance(est_report, EstimatorReport)
    assert est_report._parent_hash == cv_report._hash
    assert joblib.hash(est_report.X_train) == joblib.hash(X_experiment)
    assert joblib.hash(est_report.y_train) == joblib.hash(y_experiment)
    assert est_report.X_test is None
    assert est_report.y_test is None
    assert est_report.pos_label == cv_report.pos_label

    est_report_with_test = cv_report.create_estimator_report(
        X_test=X_heldout, y_test=y_heldout
    )

    assert isinstance(est_report_with_test, EstimatorReport)
    assert est_report_with_test._parent_hash == cv_report._hash
    assert joblib.hash(est_report_with_test.X_train) == joblib.hash(X_experiment)
    assert joblib.hash(est_report_with_test.y_train) == joblib.hash(y_experiment)
    assert joblib.hash(est_report_with_test.X_test) == joblib.hash(X_heldout)
    assert joblib.hash(est_report_with_test.y_test) == joblib.hash(y_heldout)
    assert est_report_with_test.pos_label == cv_report.pos_label


def test_pos_label_setter_propagates_to_underlying_estimator_reports(
    forest_binary_classification_data,
):
    """Check that setting pos_label on CrossValidationReport updates the report, all
    underlying estimator reports, and re-initializes internal state (hash and RNG)."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)

    assert report.pos_label is None

    report.metrics.summarize()  # trigger cache usage
    hash_before = report._hash
    rng_id_before = id(report._rng)

    report.pos_label = 1
    assert report.pos_label == 1
    for estimator_report in report.estimator_reports_:
        assert estimator_report.pos_label == 1
    assert report._hash != hash_before
    assert id(report._rng) != rng_id_before

import joblib
import numpy as np
import pandas as pd
import pytest
import skrub
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
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

    for parameter in ["estimator", "X", "y", "splitter"]:
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
        # expected n keys:
        # (result + time for 'predict'
        #  & result for 'predict_proba'/'predict_log_proba' or 'decision_function')
        # x train, test
        ("forest_binary_classification_data", 8),
        ("svc_binary_classification_data", 6),
        ("forest_multiclass_classification_data", 8),
        ("linear_regression_data", 4),
    ],
)
@pytest.mark.parametrize("n_jobs", [None, 1, 2])
def test_cache_predictions(request, fixture_name, expected_n_keys, n_jobs):
    """Check that :meth:`CrossValidationReport._cache_predictions` fills the cache."""
    estimator, X, y = request.getfixturevalue(fixture_name)
    report = CrossValidationReport(estimator, X, y, splitter=2, n_jobs=n_jobs)
    for estimator_report in report.estimator_reports_:
        assert estimator_report._cache == {}

    report._cache_predictions()

    for estimator_report in report.estimator_reports_:
        assert len(estimator_report._cache) == expected_n_keys

    report._clear_cache()
    for estimator_report in report.estimator_reports_:
        assert estimator_report._cache == {}


@pytest.mark.parametrize("data_source", ["train", "test"])
@pytest.mark.parametrize(
    "response_method", ["predict", "predict_proba", "decision_function"]
)
@pytest.mark.parametrize("pos_label", [None, 0, 1])
def test_get_predictions(
    data_source, response_method, pos_label, logistic_binary_classification_data
):
    """Check the behaviour of the `get_predictions` method."""
    estimator, X, y = logistic_binary_classification_data
    report = CrossValidationReport(
        estimator,
        X,
        y,
        splitter=2,
        pos_label=pos_label,
    )

    predictions = report.get_predictions(
        data_source=data_source,
        response_method=response_method,
    )
    assert len(predictions) == 2
    for split_idx, split_predictions in enumerate(predictions):
        if data_source == "train":
            expected_len = len(report.estimator_reports_[split_idx].y_train)
        else:
            assert data_source == "test"
            expected_len = len(report.estimator_reports_[split_idx].y_test)
        if response_method == "predict_proba" and pos_label is None:
            assert split_predictions.shape == (expected_len, 2)
        else:
            assert split_predictions.shape == (expected_len,)


def test_get_predictions_error(
    logistic_binary_classification_data,
):
    """Check that we raise an error when the data source is invalid."""
    estimator, X, y = logistic_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)

    with pytest.raises(ValueError, match="Invalid data source"):
        report.get_predictions(data_source="invalid")


def test_pickle(tmp_path, logistic_binary_classification_data):
    """Check that we can pickle an cross-validation report.

    In particular, the progress bar from rich are pickable, therefore we trigger
    the progress bar to be able to test that the progress bar is pickable.
    """
    estimator, X, y = logistic_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)
    report._cache_predictions()
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
        match="Clustering models are not supported yet. "
        "Please use a classification or regression model instead.",
    ):
        CrossValidationReport(KMeans(), X=np.random.rand(10, 5), y=None)


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

    est_report_with_test = cv_report.create_estimator_report(
        X_test=X_heldout, y_test=y_heldout
    )

    assert isinstance(est_report_with_test, EstimatorReport)
    assert joblib.hash(est_report_with_test.X_train) == joblib.hash(X_experiment)
    assert joblib.hash(est_report_with_test.y_train) == joblib.hash(y_experiment)
    assert joblib.hash(est_report_with_test.X_test) == joblib.hash(X_heldout)
    assert joblib.hash(est_report_with_test.y_test) == joblib.hash(y_heldout)
    assert est_report_with_test.pos_label == cv_report.pos_label


class _DummyClassifierBadRepr(DummyClassifier):
    def _repr_html_(self):
        raise TypeError("error")


def _assert_cross_validation_report_repr_html(
    html_out: str, expected_estimator_name: str
) -> None:
    assert "skore-cross-validation-report-" in html_out
    assert expected_estimator_name in html_out
    assert "skoreInitEstimatorReport" in html_out
    assert "report-hint-note" in html_out
    assert "docs.skore.probabl.ai" in html_out
    assert "report-tabset" in html_out
    assert "Report for" in html_out
    assert "CrossValidationReport.metrics" in html_out


def test_report_repr_html_binary_classification():
    X, y = make_classification(n_classes=2, random_state=42)
    estimator = DummyClassifier()
    report = CrossValidationReport(estimator, X, y, splitter=2)
    _assert_cross_validation_report_repr_html(report._repr_html_(), "DummyClassifier")


def test_report_repr_html_multiclass_classification(multiclass_classification_data):
    X, y = multiclass_classification_data
    estimator = DummyClassifier(strategy="uniform", random_state=0)
    report = CrossValidationReport(estimator, X, y, splitter=2)
    _assert_cross_validation_report_repr_html(report._repr_html_(), "DummyClassifier")


def test_report_repr_html_regression(regression_data):
    X, y = regression_data
    estimator = DummyRegressor()
    report = CrossValidationReport(estimator, X, y, splitter=2)
    _assert_cross_validation_report_repr_html(report._repr_html_(), "DummyRegressor")


def test_report_repr_html_multioutput_regression(regression_multioutput_data):
    X, y = regression_multioutput_data
    estimator = DummyRegressor()
    report = CrossValidationReport(estimator, X, y, splitter=2)
    _assert_cross_validation_report_repr_html(report._repr_html_(), "DummyRegressor")


@pytest.mark.parametrize("splitter", [2, 3])
def test_report_repr_html_sklearn_estimator_bad_html_repr(splitter):
    """HTML repr must still work when the underlying estimator rejects
    ``_repr_html_``."""
    X, y = make_classification(n_classes=2, random_state=42)
    estimator = _DummyClassifierBadRepr()
    report = CrossValidationReport(estimator, X, y, splitter=splitter)
    _assert_cross_validation_report_repr_html(report._repr_html_(), "DummyClassifier")


def test_report_with_data_op():
    X_a, y_a = make_classification(n_samples=10)
    data_op = skrub.X(X_a).skb.apply(LogisticRegression(), y=skrub.y(y_a))
    learner = data_op.skb.make_learner()

    report = CrossValidationReport(learner, data=data_op.skb.get_data())
    assert list(report.metrics.accuracy(aggregate="mean").columns) == [
        ("SkrubLearner", "mean")
    ]

    report = CrossValidationReport(data_op)
    assert list(report.metrics.accuracy(aggregate="mean").columns) == [
        ("SkrubLearner", "mean")
    ]


def test_from_state_bypasses_init_and_restores_state(
    monkeypatch, logistic_binary_classification_data
):
    estimator, X, y = logistic_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)
    expected_accuracy = report.metrics.accuracy()
    report._cache_predictions()
    state = report.get_state()

    def _unexpected_init(self, *args, **kwargs):
        raise AssertionError("__init__ should not be called by from_state")

    monkeypatch.setattr(CrossValidationReport, "__init__", _unexpected_init)

    restored = CrossValidationReport.from_state(state)

    assert restored.id == report.id
    assert restored.X is report.X
    assert restored.y is report.y
    assert restored.ml_task == report.ml_task
    assert restored.pos_label == report.pos_label
    assert restored._split_indices == report._split_indices
    assert len(restored.estimator_reports_) == len(report.estimator_reports_)
    assert restored.estimator_reports_[0]._cache == report.estimator_reports_[0]._cache
    assert state["estimator_reports"][0]["predictions"]
    assert state["estimator_reports"][0]["optional"]["cache"]
    assert restored.metrics.accuracy().equals(expected_accuracy)

    # check new metrics/predictions can be computed:
    restored.metrics.roc_auc()
    _ = report.get_predictions(data_source="test")
    report._cache_predictions()


def test_get_from_state_with_complex_data_op():
    X, y = make_classification(random_state=0)
    left = pd.DataFrame(
        {
            "row_id": np.arange(len(y)),
            "x0": X[:, 0],
            "x1": X[:, 1],
        }
    )
    right = pd.DataFrame(
        {
            "row_id": np.arange(len(y)),
            "x2": X[:, 2],
            "x3": X[:, 3],
        }
    )

    def _join_features(left, right):
        return left.merge(right, on="row_id").drop(columns="row_id")

    data_op = (
        skrub.var("left", left)
        .skb.apply_func(_join_features, skrub.var("right", right))
        .skb.mark_as_X()
        .skb.apply(LogisticRegression(), y=skrub.y(y))
    )
    report = CrossValidationReport(data_op, splitter=2)
    expected_accuracy = report.metrics.accuracy()
    expected_preds = report.get_predictions(data_source="test")
    state = report.get_state()

    restored = CrossValidationReport.from_state(state)

    # check fresh computations still work after restoring from state:
    restored._clear_cache()
    assert restored.metrics.accuracy().equals(expected_accuracy)
    preds = restored.get_predictions(data_source="test")
    for pred, expected_pred in zip(preds, expected_preds, strict=True):
        np.testing.assert_array_equal(pred, expected_pred)


def test_from_state_rejects_unknown_version(logistic_binary_classification_data):
    estimator, X, y = logistic_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)
    state = report.get_state() | {"version": 999}

    with pytest.raises(ValueError, match="Unexpected state version"):
        CrossValidationReport.from_state(state)

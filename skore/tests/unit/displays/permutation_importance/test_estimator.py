import matplotlib as mpl
import pandas as pd
import pytest
import skrub
from matplotlib.figure import Figure
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    make_scorer,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skore import CrossValidationReport, EstimatorReport
from skore._utils.testing import custom_r2_score


@pytest.mark.parametrize(
    "task",
    [
        "binary_classification",
        "multiclass_classification",
        "regression",
        "multioutput_regression",
    ],
)
def test_invalid_subplot_by(task, request):
    report = request.getfixturevalue(f"estimator_reports_{task}")[0]
    err_msg = "The column 'invalid' is not available for subplotting."
    display = report.inspection.permutation_importance(n_repeats=2, seed=0)
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="invalid")


@pytest.mark.parametrize(
    "task",
    [
        "binary_classification",
        "multiclass_classification",
        "regression",
        "multioutput_regression",
    ],
)
@pytest.mark.parametrize(
    "subplot_by, expected_len",
    [
        (None, 1),
        ("auto", 1),
    ],
)
def test_valid_subplot_by(task, subplot_by, expected_len, request):
    report = request.getfixturevalue(f"estimator_reports_{task}")[0]
    display = report.inspection.permutation_importance(n_repeats=2, seed=0)
    fig = display.plot(subplot_by=subplot_by)
    axes = fig.axes
    if expected_len == 1:
        assert isinstance(axes[0], mpl.axes.Axes)
    else:
        assert len(axes) == expected_len


@pytest.mark.parametrize(
    "task, metric, metric_name, subplot_by, expected_len",
    [
        (
            "multiclass_classification",
            make_scorer(precision_score, average=None),
            "precision score",
            "label",
            3,
        ),
        (
            "multioutput_regression",
            make_scorer(r2_score, multioutput="raw_values"),
            "r2 score",
            "output",
            2,
        ),
    ],
)
def test_subplot_by_non_averaged_metrics(
    task, metric, metric_name, subplot_by, expected_len, request
):
    report = request.getfixturevalue(f"estimator_reports_{task}")[0]
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=metric
    )
    fig = display.plot(metric=metric_name, subplot_by=subplot_by)
    axes = fig.axes
    assert len(axes) == expected_len

    valid_values = [subplot_by, "auto", "None"]
    err_msg = (
        f"The column 'invalid' is not available for subplotting. You can use the "
        f"following values to create subplots: {', '.join(valid_values)}"
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="invalid")


def test_multiple_metrics_require_metric_param(estimator_reports_regression):
    report = estimator_reports_regression[0]
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=["r2", "neg_mean_squared_error"]
    )
    with pytest.raises(ValueError, match="Please select a metric"):
        display.plot()

    fig = display.plot(metric="r2")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Decrease in r2"
    fig = display.plot(metric="neg_mean_squared_error")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Decrease in neg_mean_squared_error"


def test_frame_metric_filter(estimator_reports_regression):
    report = estimator_reports_regression[0]
    display = report.inspection.permutation_importance(
        n_repeats=2,
        seed=0,
        metric=["r2", "neg_mean_squared_error"],
    )
    assert set(display.frame()["metric"]) == {"r2", "neg_mean_squared_error"}
    assert set(display.frame(metric="r2")["metric"]) == {"r2"}
    assert set(display.frame(metric=["r2"])["metric"]) == {"r2"}


def test_callable_metric_name(estimator_reports_regression):
    report = estimator_reports_regression[0]
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=custom_r2_score
    )
    fig = display.plot(metric="custom r2 score")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Decrease in custom r2 score"


def test_per_label_metrics_frame(
    estimator_reports_multiclass_classification,
):
    report = estimator_reports_multiclass_classification[0]
    metrics = {
        "precision": make_scorer(precision_score, average=None),
        "recall": make_scorer(recall_score, average=None),
    }
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=metrics
    )
    frame = display.frame()
    assert set(frame["label"]) == {0, 1, 2}
    assert set(frame["metric"]) == {"precision", "recall"}

    assert "output" not in frame.columns
    assert "label" in frame.columns


def test_per_output_metrics_frame(
    estimator_reports_multioutput_regression,
):
    report = estimator_reports_multioutput_regression[0]
    metric = {
        "r2": make_scorer(r2_score, multioutput="raw_values"),
        "mse": make_scorer(mean_squared_error, multioutput="raw_values"),
    }
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=metric
    )
    frame = display.frame()
    assert set(frame["output"]) == {0, 1}
    assert set(frame["metric"]) == {"r2", "mse"}

    assert "output" in frame.columns
    assert "label" not in frame.columns


@pytest.mark.parametrize("aggregate", [None, ("mean", "std")])
def test_frame_mixed_averaged_and_non_averaged_metrics(
    estimator_reports_binary_classification, aggregate
):
    report = estimator_reports_binary_classification[0]
    metrics = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average=None),
    }
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=metrics
    )
    frame = display.frame(aggregate=aggregate)

    assert set(frame["metric"]) == {"accuracy", "precision"}
    assert "label" in frame.columns
    assert frame.query("metric == 'accuracy'")["label"].isna().all()
    assert not frame.query("metric == 'precision'")["label"].isna().any()


def test_plot_mixed_averaged_and_non_averaged_metrics(
    estimator_reports_binary_classification,
):
    report = estimator_reports_binary_classification[0]
    metrics = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average=None),
    }
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=metrics
    )
    fig = display.plot(metric="accuracy")
    assert isinstance(fig, Figure)
    fig = display.plot(metric="precision")
    assert isinstance(fig, Figure)


def test_default_metric_name_classifier(estimator_reports_binary_classification):
    report = estimator_reports_binary_classification[0]
    display = report.inspection.permutation_importance(n_repeats=2, seed=0)
    assert set(display.importances["metric"]) == {"accuracy"}


def test_default_metric_name_regressor(estimator_reports_regression):
    report = estimator_reports_regression[0]
    display = report.inspection.permutation_importance(n_repeats=2, seed=0)
    assert set(display.importances["metric"]) == {"r2"}


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_data_source(estimator_reports_binary_classification, data_source):
    report = estimator_reports_binary_classification[0]
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, data_source=data_source
    )
    assert set(display.importances["data_source"]) == {data_source}


def _skrub_and_sklearn_reports():
    X, y = make_regression(n_samples=60, n_features=3, random_state=0)
    X = pd.DataFrame(X, columns=["a", "b", "c"])
    data_op = skrub.X().skb.apply(StandardScaler()).skb.apply(Ridge(), y=skrub.y())
    skrub_report = EstimatorReport(
        data_op.skb.make_learner(),
        train_data={"X": X[:40], "y": y[:40]},
        test_data={"X": X[40:], "y": y[40:]},
    )
    sklearn_report = EstimatorReport(
        make_pipeline(StandardScaler(), Ridge()),
        X_train=X[:40],
        y_train=y[:40],
        X_test=X[40:],
        y_test=y[40:],
    )
    return skrub_report, sklearn_report


@pytest.mark.parametrize("at_step", [0, -1])
@pytest.mark.parametrize("data_source", ["train", "test"])
@pytest.mark.parametrize("metric", [None, "neg_mean_squared_error"])
def test_skrub_learner_matches_sklearn_pipeline(at_step, data_source, metric):
    """A skrub learner gives the same importances as the equivalent scikit-learn
    pipeline, both on the ``X`` node features and on the predictor input."""
    skrub_report, sklearn_report = _skrub_and_sklearn_reports()
    kwargs = {
        "at_step": at_step,
        "data_source": data_source,
        "metric": metric,
        "n_repeats": 2,
        "max_samples": 0.8,
        "seed": 0,
    }
    pd.testing.assert_frame_equal(
        skrub_report.inspection.permutation_importance(**kwargs).frame(),
        sklearn_report.inspection.permutation_importance(**kwargs).frame(),
    )


@pytest.mark.parametrize("at_step", [1, -2, "ridge"])
def test_skrub_learner_invalid_at_step(at_step):
    """Only 0 and -1 are supported for skrub learners."""
    skrub_report, _ = _skrub_and_sklearn_reports()
    with pytest.raises(ValueError, match="at_step must be 0 or -1"):
        skrub_report.inspection.permutation_importance(at_step=at_step, seed=0)


def test_skrub_learner_cross_validation():
    """Permutation importance works on a cross-validation report of a skrub
    learner."""
    X, y = make_regression(n_samples=60, n_features=3, random_state=0)
    X = pd.DataFrame(X, columns=["a", "b", "c"])
    data_op = skrub.X().skb.apply(Ridge(), y=skrub.y())
    report = CrossValidationReport(
        data_op.skb.make_learner(), data={"X": X, "y": y}, splitter=2
    )
    frame = report.inspection.permutation_importance(n_repeats=2, seed=0).frame()
    assert frame["feature"].tolist() == ["a", "b", "c"]

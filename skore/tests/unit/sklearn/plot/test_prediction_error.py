import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from skore import EstimatorReport
from skore.sklearn._plot import PredictionErrorDisplay


@pytest.fixture
def regression_data():
    X, y = make_regression(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return LinearRegression().fit(X_train, y_train), X_train, X_test, y_train, y_test


@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({"subsample": -1}, "When an integer, subsample=-1 should be"),
        ({"subsample": 20.0}, "When a floating-point, subsample=20.0 should be"),
        ({"subsample": -20.0}, "When a floating-point, subsample=-20.0 should be"),
        ({"kind": "xxx"}, "`kind` must be one of"),
    ],
)
def test_prediction_error_display_raise_error(pyplot, params, err_msg, regression_data):
    """Check that we raise the proper error when making the parameters
    validation."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.plot.prediction_error(**params)


def test_prediction_error_display_regression(pyplot, regression_data):
    """Check the attributes and default plotting behaviour of the prediction error plot
    with regression data."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.plot.prediction_error()
    assert isinstance(display, PredictionErrorDisplay)

    # check the structure of the attributes
    assert isinstance(display.y_true, np.ndarray)
    assert isinstance(display.y_pred, np.ndarray)
    np.testing.assert_allclose(display.y_true, y_test)
    np.testing.assert_allclose(display.y_pred, estimator.predict(X_test))
    assert display.data_source == "test"

    # check the default plotting behaviour
    import matplotlib as mpl

    assert isinstance(display.line_, mpl.lines.Line2D)
    assert display.line_.get_label() == "Perfect predictions"
    assert display.line_.get_color() == "black"

    assert isinstance(display.scatter_, mpl.collections.PathCollection)

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == estimator.__class__.__name__
    assert len(legend.get_texts()) == 2

    assert display.ax_.get_xlabel() == "Predicted values"
    assert display.ax_.get_ylabel() == "Residuals (actual - predicted)"


def test_prediction_error_display_regression_kind(pyplot, regression_data):
    """Check the attributes when switching to the "actual_vs_predicted" kind."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.plot.prediction_error(kind="actual_vs_predicted")
    assert isinstance(display, PredictionErrorDisplay)

    import matplotlib as mpl

    assert isinstance(display.line_, mpl.lines.Line2D)
    assert display.line_.get_label() == "Perfect predictions"
    assert display.line_.get_color() == "black"

    assert isinstance(display.scatter_, mpl.collections.PathCollection)

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == estimator.__class__.__name__
    assert len(legend.get_texts()) == 2

    assert display.ax_.get_xlabel() == "Predicted values"
    assert display.ax_.get_ylabel() == "Actual values"

    assert display.ax_.get_xlim() == display.ax_.get_ylim()
    assert display.ax_.get_aspect() in ("equal", 1.0)


def test_prediction_error_display_data_source(pyplot, regression_data):
    """Check that we can pass the `data_source` argument to the prediction error
    plot."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.plot.prediction_error(data_source="train")
    assert display.line_.get_label() == "Perfect predictions"
    assert display.scatter_.get_label() == "Train set"

    display = report.metrics.plot.prediction_error(
        data_source="X_y", X=X_train, y=y_train
    )
    assert display.line_.get_label() == "Perfect predictions"
    assert display.scatter_.get_label() == "Data set"


def test_prediction_error_display_kwargs(pyplot, regression_data):
    """Check that we can pass keyword arguments to the prediction error plot."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.plot.prediction_error()
    display.plot(scatter_kwargs={"color": "red"}, line_kwargs={"color": "blue"})
    np.testing.assert_allclose(display.scatter_.get_facecolor(), [[1, 0, 0, 0.8]])
    assert display.line_.get_color() == "blue"

    display.plot(despine=False)
    assert display.ax_.spines["top"].get_visible()
    assert display.ax_.spines["right"].get_visible()

    expected_subsample = 10
    display = report.metrics.plot.prediction_error(subsample=expected_subsample)
    assert len(display.scatter_.get_offsets()) == expected_subsample

    expected_subsample = int(X_test.shape[0] * 0.5)
    display = report.metrics.plot.prediction_error(subsample=0.5)
    assert len(display.scatter_.get_offsets()) == expected_subsample

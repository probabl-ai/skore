import matplotlib as mpl
import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from skore import CrossValidationReport, EstimatorReport
from skore.sklearn._plot import PredictionErrorDisplay


@pytest.fixture
def regression_data():
    X, y = make_regression(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return LinearRegression().fit(X_train, y_train), X_train, X_test, y_train, y_test


@pytest.fixture
def regression_data_no_split():
    X, y = make_regression(random_state=42)
    return LinearRegression(), X, y


@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({"subsample": -1}, "When an integer, subsample=-1 should be"),
        ({"subsample": 20.0}, "When a floating-point, subsample=20.0 should be"),
        ({"subsample": -20.0}, "When a floating-point, subsample=-20.0 should be"),
    ],
)
def test_prediction_error_display_raise_error(pyplot, params, err_msg, regression_data):
    """
    Check that we raise the proper error when making the parameters
    validation."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.prediction_error(**params)


@pytest.mark.parametrize("subsample", [None, 1_000])
def test_prediction_error_display_regression(pyplot, regression_data, subsample):
    """
    Check the attributes and default plotting behaviour of the prediction error plot
    with regression data."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.prediction_error(subsample=subsample)
    assert isinstance(display, PredictionErrorDisplay)

    # check the structure of the attributes
    assert isinstance(display.y_true, list)
    assert isinstance(display.y_pred, list)
    np.testing.assert_allclose(display.y_true[0], y_test)
    np.testing.assert_allclose(display.y_pred[0], estimator.predict(X_test))
    assert display.data_source == "test"

    display.plot()
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

    assert display.ax_.get_aspect() not in ("equal", 1.0)


def test_prediction_error_cross_validation_display_regression(
    pyplot, regression_data_no_split
):
    """
    Check the attributes and default plotting behaviour of the prediction error plot
    with cross-validation data."""
    (estimator, X, y), cv = regression_data_no_split, 3
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.prediction_error()
    assert isinstance(display, PredictionErrorDisplay)

    # check the structure of the attributes
    assert isinstance(display.y_true, list)
    assert isinstance(display.y_pred, list)
    assert len(display.y_true) == len(display.y_pred) == cv
    assert display.data_source == "test"

    display.plot()
    assert isinstance(display.line_, mpl.lines.Line2D)
    assert display.line_.get_label() == "Perfect predictions"
    assert display.line_.get_color() == "black"

    assert isinstance(display.scatter_, mpl.collections.PathCollection)

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == estimator.__class__.__name__
    assert len(legend.get_texts()) == 4

    assert display.ax_.get_xlabel() == "Predicted values"
    assert display.ax_.get_ylabel() == "Residuals (actual - predicted)"

    assert display.ax_.get_aspect() not in ("equal", 1.0)


def test_prediction_error_display_regression_kind(pyplot, regression_data):
    """Check the attributes when switching to the "actual_vs_predicted" kind."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.prediction_error()
    assert isinstance(display, PredictionErrorDisplay)

    display.plot(kind="actual_vs_predicted")
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


def test_prediction_error_cross_validation_display_regression_kind(
    pyplot, regression_data_no_split
):
    """Check the attributes when switching to the "actual_vs_predicted" kind."""
    (estimator, X, y), cv = regression_data_no_split, 3
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.prediction_error()
    display.plot(kind="actual_vs_predicted")
    assert isinstance(display, PredictionErrorDisplay)

    # check the structure of the attributes
    assert isinstance(display.y_true, list)
    assert isinstance(display.y_pred, list)
    assert len(display.y_true) == len(display.y_pred) == cv
    assert display.data_source == "test"

    assert isinstance(display.line_, mpl.lines.Line2D)
    assert display.line_.get_label() == "Perfect predictions"
    assert display.line_.get_color() == "black"

    assert isinstance(display.scatter_, mpl.collections.PathCollection)

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == estimator.__class__.__name__
    assert len(legend.get_texts()) == 4

    assert display.ax_.get_xlabel() == "Predicted values"
    assert display.ax_.get_ylabel() == "Actual values"

    assert display.ax_.get_aspect() in ("equal", 1.0)


def test_prediction_error_display_data_source(pyplot, regression_data):
    """
    Check that we can pass the `data_source` argument to the prediction error
    plot."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.prediction_error(data_source="train")
    display.plot()
    assert display.line_.get_label() == "Perfect predictions"
    assert display.scatter_.get_label() == "Train set"

    display = report.metrics.prediction_error(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    assert display.line_.get_label() == "Perfect predictions"
    assert display.scatter_.get_label() == "Data set"


def test_prediction_error_display_kwargs(pyplot, regression_data):
    """Check that we can pass keyword arguments to the prediction error plot."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.prediction_error()
    display.plot(scatter_kwargs={"color": "red"}, line_kwargs={"color": "blue"})
    np.testing.assert_allclose(display.scatter_.get_facecolor(), [[1, 0, 0, 0.3]])
    assert display.line_.get_color() == "blue"

    # check the `.style` display setter
    display.plot()  # default style
    np.testing.assert_allclose(
        display.scatter_.get_facecolor(),
        [[0.121569, 0.466667, 0.705882, 0.3]],
        rtol=1e-3,
    )
    assert display.line_.get_color() == "black"
    display.set_style(scatter_kwargs={"color": "red"}, line_kwargs={"color": "blue"})
    display.plot()
    np.testing.assert_allclose(display.scatter_.get_facecolor(), [[1, 0, 0, 0.3]])
    assert display.line_.get_color() == "blue"
    # overwrite the style that was set above
    display.plot(
        scatter_kwargs={"color": "tab:orange"}, line_kwargs={"color": "tab:green"}
    )
    np.testing.assert_allclose(
        display.scatter_.get_facecolor(), [[1.0, 0.498039, 0.054902, 0.3]], rtol=1e-3
    )
    assert display.line_.get_color() == "tab:green"

    display.plot(despine=False)
    assert display.ax_.spines["top"].get_visible()
    assert display.ax_.spines["right"].get_visible()

    expected_subsample = 10
    display = report.metrics.prediction_error(subsample=expected_subsample)
    display.plot()
    assert len(display.scatter_.get_offsets()) == expected_subsample

    expected_subsample = int(X_test.shape[0] * 0.5)
    display = report.metrics.prediction_error(subsample=0.5)
    display.plot()
    assert len(display.scatter_.get_offsets()) == expected_subsample

import matplotlib as mpl
import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from skore import ComparisonReport, CrossValidationReport, EstimatorReport
from skore.sklearn._plot import PredictionErrorDisplay
from skore.sklearn._plot.metrics.prediction_error import RangeData


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
    """Check that we raise the proper error when making the parameters
    validation."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.prediction_error(**params)


@pytest.mark.parametrize("subsample", [None, 1_000])
def test_prediction_error_display_regression(pyplot, regression_data, subsample):
    """Check the attributes and default plotting behaviour of the prediction error plot
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
    assert isinstance(display.residuals, list)
    np.testing.assert_allclose(display.y_true[0], y_test)
    np.testing.assert_allclose(display.y_pred[0], estimator.predict(X_test))
    np.testing.assert_allclose(display.residuals[0], y_test - estimator.predict(X_test))
    assert display.data_source == "test"
    assert isinstance(display.range_y_true, RangeData)
    assert isinstance(display.range_y_pred, RangeData)
    assert isinstance(display.range_residuals, RangeData)
    assert display.range_y_true.min == np.min(display.y_true[0])
    assert display.range_y_true.max == np.max(display.y_true[0])
    assert display.range_y_pred.min == np.min(display.y_pred[0])
    assert display.range_y_pred.max == np.max(display.y_pred[0])
    assert display.range_residuals.min == np.min(display.residuals[0])
    assert display.range_residuals.max == np.max(display.residuals[0])

    display.plot()
    assert hasattr(display, "ax_")
    assert hasattr(display, "figure_")
    assert isinstance(display.line_, mpl.lines.Line2D)
    assert display.line_.get_label() == "Perfect predictions"
    assert display.line_.get_color() == "black"

    assert isinstance(display.scatter_, list)
    for scatter in display.scatter_:
        assert isinstance(scatter, mpl.collections.PathCollection)

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == "LinearRegression"
    assert len(legend.get_texts()) == 2

    assert display.ax_.get_xlabel() == "Predicted values"
    assert display.ax_.get_ylabel() == "Residuals (actual - predicted)"

    assert display.ax_.get_aspect() not in ("equal", 1.0)


@pytest.mark.parametrize("data_source", ["train", "test", "X_y"])
def test_prediction_error_cross_validation_display_regression(
    pyplot, regression_data_no_split, data_source
):
    """Check the attributes and default plotting behaviour of the prediction error plot
    with cross-validation data."""
    (estimator, X, y), cv = regression_data_no_split, 3
    if data_source == "X_y":
        prediction_error_kwargs = {"data_source": data_source, "X": X, "y": y}
    else:
        prediction_error_kwargs = {"data_source": data_source}

    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.prediction_error(**prediction_error_kwargs)
    assert isinstance(display, PredictionErrorDisplay)

    # check the structure of the attributes
    assert isinstance(display.y_true, list)
    assert isinstance(display.y_pred, list)
    assert isinstance(display.residuals, list)
    assert len(display.y_true) == len(display.y_pred) == len(display.residuals) == cv
    assert display.data_source == data_source
    assert isinstance(display.range_y_true, RangeData)
    assert isinstance(display.range_y_pred, RangeData)
    assert isinstance(display.range_residuals, RangeData)
    for attr in ("y_true", "y_pred", "residuals"):
        global_min, global_max = np.inf, -np.inf
        for display_attr in getattr(display, attr):
            global_min = min(global_min, np.min(display_attr))
            global_max = max(global_max, np.max(display_attr))
        assert getattr(display, f"range_{attr}").min == global_min
        assert getattr(display, f"range_{attr}").max == global_max

    display.plot()
    assert isinstance(display.line_, mpl.lines.Line2D)
    assert display.line_.get_label() == "Perfect predictions"
    assert display.line_.get_color() == "black"

    assert isinstance(display.scatter_, list)
    for scatter in display.scatter_:
        assert isinstance(scatter, mpl.collections.PathCollection)

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    data_source_title = "external" if data_source == "X_y" else data_source
    assert (
        legend.get_title().get_text()
        == f"LinearRegression on $\\bf{{{data_source_title}}}$ set"
    )
    assert len(legend.get_texts()) == 4

    assert display.ax_.get_xlabel() == "Predicted values"
    assert display.ax_.get_ylabel() == "Residuals (actual - predicted)"

    assert display.ax_.get_aspect() not in ("equal", 1.0)


def test_prediction_error_comparison_report_display_regression(pyplot, regression_data):
    """Check the attributes and default plotting behaviour of the prediction error plot
    with a comparison report."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = ComparisonReport(
        reports={
            "estimator 1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator 2": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        },
    )
    display = report.metrics.prediction_error()
    assert isinstance(display, PredictionErrorDisplay)

    # check the structure of the attributes
    assert display.estimator_names == ["estimator 1", "estimator 2"]
    assert isinstance(display.y_true, list)
    assert isinstance(display.y_pred, list)
    assert isinstance(display.residuals, list)
    assert len(display.y_true) == len(display.y_pred) == len(display.residuals) == 2
    assert display.data_source == "test"
    assert isinstance(display.range_y_true, RangeData)
    assert isinstance(display.range_y_pred, RangeData)
    assert isinstance(display.range_residuals, RangeData)
    for attr in ("y_true", "y_pred", "residuals"):
        global_min, global_max = np.inf, -np.inf
        for display_attr in getattr(display, attr):
            global_min = min(global_min, np.min(display_attr))
            global_max = max(global_max, np.max(display_attr))
        assert getattr(display, f"range_{attr}").min == global_min
        assert getattr(display, f"range_{attr}").max == global_max

    display.plot()
    assert isinstance(display.line_, mpl.lines.Line2D)
    assert display.line_.get_label() == "Perfect predictions"
    assert display.line_.get_color() == "black"

    assert isinstance(display.scatter_, list)
    for scatter in display.scatter_:
        assert isinstance(scatter, mpl.collections.PathCollection)

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == "Prediction errors on $\\bf{test}$ set"
    assert len(legend.get_texts()) == 3

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

    assert isinstance(display.scatter_, list)
    for scatter in display.scatter_:
        assert isinstance(scatter, mpl.collections.PathCollection)

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

    assert isinstance(display.scatter_, list)
    for scatter in display.scatter_:
        assert isinstance(scatter, mpl.collections.PathCollection)

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == "LinearRegression on $\\bf{test}$ set"
    assert len(legend.get_texts()) == 4

    assert display.ax_.get_xlabel() == "Predicted values"
    assert display.ax_.get_ylabel() == "Actual values"

    assert display.ax_.get_aspect() in ("equal", 1.0)


def test_prediction_error_comparison_report_display_regression_kind(
    pyplot, regression_data
):
    """Check the attributes when switching to the "actual_vs_predicted" kind."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = ComparisonReport(
        reports={
            "estimator 1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator 2": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        },
    )
    display = report.metrics.prediction_error()
    display.plot(kind="actual_vs_predicted")
    assert isinstance(display, PredictionErrorDisplay)

    # check the structure of the attributes
    assert isinstance(display.y_true, list)
    assert isinstance(display.y_pred, list)
    assert len(display.y_true) == len(display.y_pred) == 2
    assert display.data_source == "test"

    assert isinstance(display.line_, mpl.lines.Line2D)
    assert display.line_.get_label() == "Perfect predictions"
    assert display.line_.get_color() == "black"

    assert isinstance(display.scatter_, list)
    for scatter in display.scatter_:
        assert isinstance(scatter, mpl.collections.PathCollection)

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == "Prediction errors on $\\bf{test}$ set"
    assert len(legend.get_texts()) == 3

    assert display.ax_.get_xlabel() == "Predicted values"
    assert display.ax_.get_ylabel() == "Actual values"

    assert display.ax_.get_aspect() in ("equal", 1.0)


def test_prediction_error_display_data_source(pyplot, regression_data):
    """Check that we can pass the `data_source` argument to the prediction error
    plot."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.prediction_error(data_source="train")
    display.plot()
    assert display.line_.get_label() == "Perfect predictions"
    assert display.scatter_[0].get_label() == "Train set"

    display = report.metrics.prediction_error(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    assert display.line_.get_label() == "Perfect predictions"
    assert display.scatter_[0].get_label() == "Data set"


def test_prediction_error_display_kwargs(pyplot, regression_data):
    """Check that we can pass keyword arguments to the prediction error plot."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.prediction_error()
    display.plot(
        data_points_kwargs={"color": "red"}, perfect_model_kwargs={"color": "blue"}
    )
    np.testing.assert_allclose(display.scatter_[0].get_facecolor(), [[1, 0, 0, 0.3]])
    assert display.line_.get_color() == "blue"

    # check the `.style` display setter
    display.plot()  # default style
    np.testing.assert_allclose(
        display.scatter_[0].get_facecolor(),
        [[0.121569, 0.466667, 0.705882, 0.3]],
        rtol=1e-3,
    )
    assert display.line_.get_color() == "black"
    display.set_style(
        data_points_kwargs={"color": "red"}, perfect_model_kwargs={"color": "blue"}
    )
    display.plot()
    np.testing.assert_allclose(display.scatter_[0].get_facecolor(), [[1, 0, 0, 0.3]])
    assert display.line_.get_color() == "blue"
    # overwrite the style that was set above
    display.plot(
        data_points_kwargs={"color": "tab:orange"},
        perfect_model_kwargs={"color": "tab:green"},
    )
    np.testing.assert_allclose(
        display.scatter_[0].get_facecolor(),
        [[1.0, 0.498039, 0.054902, 0.3]],
        rtol=1e-3,
    )
    assert display.line_.get_color() == "tab:green"

    display.plot(despine=False)
    assert display.ax_.spines["top"].get_visible()
    assert display.ax_.spines["right"].get_visible()

    expected_subsample = 10
    display = report.metrics.prediction_error(subsample=expected_subsample)
    display.plot()
    assert len(display.scatter_[0].get_offsets()) == expected_subsample

    expected_subsample = int(X_test.shape[0] * 0.5)
    display = report.metrics.prediction_error(subsample=0.5)
    display.plot()
    assert len(display.scatter_[0].get_offsets()) == expected_subsample


def test_prediction_error_display_cross_validation_kwargs(
    pyplot, regression_data_no_split
):
    """Check that we can pass keyword arguments to the prediction error plot when
    there is a cross-validation report."""
    (estimator, X, y), cv = regression_data_no_split, 3
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.prediction_error()
    display.plot(
        data_points_kwargs=[{"color": "red"}, {"color": "green"}, {"color": "blue"}],
        perfect_model_kwargs={"color": "orange"},
    )
    rgb_colors = [
        [[1.0, 0.0, 0.0, 0.3]],
        [[0.0, 0.50196078, 0.0, 0.3]],
        [[0.0, 0.0, 1.0, 0.3]],
    ]
    for scatter, rgb_color in zip(display.scatter_, rgb_colors):
        np.testing.assert_allclose(scatter.get_facecolor(), rgb_color, rtol=1e-3)
    assert display.line_.get_color() == "orange"


def test_prediction_error_display_comparison_estimator_kwargs(pyplot, regression_data):
    """Check that we can pass keyword arguments to the prediction error plot when
    there is a comparison report."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = ComparisonReport(
        reports={
            "estimator 1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator 2": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        },
    )
    display = report.metrics.prediction_error()
    display.plot(
        data_points_kwargs=[{"color": "red"}, {"color": "blue"}],
        perfect_model_kwargs={"color": "orange"},
    )
    rgb_colors = [[[1.0, 0.0, 0.0, 0.3]], [[0.0, 0.0, 1.0, 0.3]]]
    for scatter, rgb_color in zip(display.scatter_, rgb_colors):
        np.testing.assert_allclose(scatter.get_facecolor(), rgb_color, rtol=1e-3)
    assert display.line_.get_color() == "orange"


def test_random_state(regression_data):
    """If random_state is None (the default) the call should not be cached."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report.metrics.prediction_error()
    # skore should store the y_pred, but not the plot
    assert len(report._cache) == 2


@pytest.mark.parametrize("data_points_kwargs", ["not a dict", [{"color": "red"}]])
def test_prediction_error_single_estimator_kwargs_error(
    pyplot, regression_data, data_points_kwargs
):
    """Check that we raise an error when we pass keyword arguments to the prediction
    error plot if there is a single estimator."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.prediction_error()

    err_msg = (
        "You intend to plot the prediction error for a single estimator. We expect "
        "`data_points_kwargs` to be a dictionary."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(data_points_kwargs=data_points_kwargs)


@pytest.mark.parametrize("data_points_kwargs", ["not a list", [{"color": "red"}]])
def test_prediction_error_cross_validation_kwargs_error(
    pyplot, regression_data_no_split, data_points_kwargs
):
    """Check that we raise an error when we pass keyword arguments to the prediction
    error plot if there is a cross-validation report."""
    (estimator, X, y), cv = regression_data_no_split, 3
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.prediction_error()

    err_msg = (
        "You intend to plot prediction errors either from multiple estimators "
        "or from a cross-validated estimator. We expect `data_points_kwargs` to be "
        "a list of dictionaries with the same length as the number of "
        "estimators or splits."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(data_points_kwargs=data_points_kwargs)


@pytest.mark.parametrize("data_points_kwargs", ["not a list", [{"color": "red"}]])
def test_prediction_error_comparison_estimator_kwargs_error(
    pyplot, regression_data, data_points_kwargs
):
    """Check that we raise an error when we pass keyword arguments to the prediction
    error plot if there is a comparison report."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = ComparisonReport(
        reports={
            "estimator 1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator 2": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        },
    )
    display = report.metrics.prediction_error()

    err_msg = (
        "You intend to plot prediction errors either from multiple estimators "
        "or from a cross-validated estimator. We expect `data_points_kwargs` to be "
        "a list of dictionaries with the same length as the number of "
        "estimators or splits."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(data_points_kwargs=data_points_kwargs)


def test_prediction_error_display_error_kind(pyplot, regression_data):
    """Check that we raise an error when we pass the `kind` argument to the prediction
    error plot. Since all reports shares the same `plot` method, we don't need to check
    all types of reports."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.prediction_error()

    err_msg = (
        "`kind` must be one of actual_vs_predicted, residual_vs_predicted. Got "
        "'whatever' instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(kind="whatever")


def test_prediction_error_display_wrong_report_type(pyplot, regression_data):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `report_type` argument."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    estimator_report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = estimator_report.metrics.prediction_error()
    display.report_type = "unknown"
    err_msg = (
        "`report_type` should be one of 'estimator', 'cross-validation', "
        "or 'comparison-estimator'. Got 'unknown' instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot()

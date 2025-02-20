from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from numpy import array, array_equal
from numpy.testing import assert_equal
from pytest import fixture
from skore.sklearn._comparison.prediction_error_display import PredictionErrorDisplay


@fixture
def regression_display():
    y_true = (array((-1, 0, 1)), array((-1, 0, 1)))
    y_pred = (array((0, 0.2, 0.8)), array((0.8, 0.2, 0)))

    return PredictionErrorDisplay._from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        estimator_names=["R-E1", "R-E2"],
        ml_task="regression",
        data_source="test",
    )


class TestPredictionErrorDisplay:
    def test_from_predictions(self, regression_display):
        display = regression_display

        assert_equal(display.y_true, [(-1, 0, 1), (-1, 0, 1)])
        assert_equal(display.y_pred, [(0, 0.2, 0.8), (0.8, 0.2, 0)])
        assert display.estimator_names == ["R-E1", "R-E2"]
        assert display.data_source == "test"

    def test_plot_residual_vs_predicted(self, tmp_path, regression_display):
        display = regression_display
        display.plot(kind="residual_vs_predicted")

        # Test `line_` attribute
        assert isinstance(display.line_, Line2D)
        assert_equal(display.line_.get_xdata(), (0.0, 0.8))
        assert_equal(display.line_.get_ydata(), (0.0, 0.0))
        assert display.line_.get_label() == "Perfect predictions"
        assert display.line_.get_color() == "black"

        # Test `scatters_` attribute
        assert isinstance(display.scatters_, list)
        assert len(display.scatters_) == 2
        assert all(isinstance(scatter, PathCollection) for scatter in display.scatters_)
        assert_equal(
            display.scatters_[0].get_offsets().data,
            ((0.0, (-1.0 - 0)), (0.2, (0.0 - 0.2)), (0.8, (1 - 0.8))),
        )
        assert_equal(
            display.scatters_[1].get_offsets().data,
            ((0.8, (-1 - 0.8)), (0.2, (0 - 0.2)), (0.0, (1.0 - 0.0))),
        )
        assert display.scatters_[0].get_label() == "R-E1 #1"
        assert display.scatters_[1].get_label() == "R-E2 #2"
        assert not array_equal(
            display.scatters_[0].get_facecolor(), display.scatters_[1].get_facecolor()
        )

        # Test `ax_` attribute, its scatters, legend and title
        assert isinstance(display.ax_, Axes)
        assert display.ax_.collections[:2] == display.scatters_
        assert (
            display.ax_.get_legend().get_title().get_text()
            == "Regression on $\\bf{test}$ set"
        )
        assert display.ax_.get_xlabel() == "Predicted values"
        assert display.ax_.get_ylabel() == "Residuals (actual - predicted)"
        assert display.ax_.get_adjustable() == "box"
        assert display.ax_.get_aspect() in ("equal", 1.0)
        assert display.ax_.get_xlim() == (0.00, 0.80)
        assert display.ax_.get_ylim() == (-1.80, 1.00)

        # Test `figure_` attribute
        assert display.ax_.figure == display.figure_

    def test_plot_actual_vs_predicted(self, tmp_path, regression_display):
        display = regression_display
        display.plot(kind="actual_vs_predicted")

        # Test `line_` attribute
        assert isinstance(display.line_, Line2D)
        assert_equal(display.line_.get_xdata(), (-1.0, 1.0))
        assert_equal(display.line_.get_ydata(), (-1.0, 1.0))
        assert display.line_.get_label() == "Perfect predictions"
        assert display.line_.get_color() == "black"

        # Test `scatters_` attribute
        assert isinstance(display.scatters_, list)
        assert len(display.scatters_) == 2
        assert all(isinstance(scatter, PathCollection) for scatter in display.scatters_)
        assert_equal(
            display.scatters_[0].get_offsets().data,
            ((0.0, -1.0), (0.2, 0.0), (0.8, 1.0)),
        )
        assert_equal(
            display.scatters_[1].get_offsets().data,
            ((0.8, -1.0), (0.2, 0.0), (0.0, 1.0)),
        )
        assert display.scatters_[0].get_label() == "R-E1 #1"
        assert display.scatters_[1].get_label() == "R-E2 #2"
        assert not array_equal(
            display.scatters_[0].get_facecolor(), display.scatters_[1].get_facecolor()
        )

        # Test `ax_` attribute, its scatters, legend and title
        assert isinstance(display.ax_, Axes)
        assert display.ax_.collections[:2] == display.scatters_
        assert (
            display.ax_.get_legend().get_title().get_text()
            == "Regression on $\\bf{test}$ set"
        )
        assert display.ax_.get_xlabel() == "Predicted values"
        assert display.ax_.get_ylabel() == "Actual values"
        assert display.ax_.get_adjustable() == "box"
        assert display.ax_.get_aspect() in ("equal", 1.0)
        assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-1.00, 1.00)

        # Test `figure_` attribute
        assert display.ax_.figure == display.figure_

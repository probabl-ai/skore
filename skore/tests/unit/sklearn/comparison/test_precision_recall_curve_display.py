from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy import array
from numpy.testing import assert_equal
from pytest import fixture
from sklearn.dummy import DummyClassifier
from skore.sklearn._comparison.precision_recall_curve_display import (
    PrecisionRecallCurveDisplay,
)


@fixture
def binary_classification_display():
    y_true = (array((0, 1)), array((0, 1)))
    y_pred = (array((0.2, 0.8)), array((0.8, 0.2)))
    estimators = (
        DummyClassifier().fit((0, 1), (0, 1)),
        DummyClassifier().fit((0, 1), (0, 1)),
    )

    return PrecisionRecallCurveDisplay._from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        estimators=estimators,
        estimator_names=["BC-E1", "BC-E2"],
        ml_task="binary-classification",
        data_source="test",
    )


@fixture
def multiclass_classification_display():
    y_true = (array((0, 1, 2)), array((0, 1, 2)))
    y_pred = (
        array(((0.8, 0.2, 0.0), (0.0, 0.8, 0.2), (0.2, 0.0, 0.8))),
        array(((0.0, 0.2, 0.8), (0.8, 0.0, 0.2), (0.2, 0.8, 0.0))),
    )
    estimators = (
        DummyClassifier().fit((0, 1, 2), (0, 1, 2)),
        DummyClassifier().fit((0, 1, 2), (0, 1, 2)),
    )

    return PrecisionRecallCurveDisplay._from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        estimators=estimators,
        estimator_names=["MC-E1", "MC-E2"],
        ml_task="multiclass-classification",
        data_source="test",
    )


class TestPrecisionRecallCurveDisplay:
    def test_from_predictions_binary_classification(
        self, binary_classification_display
    ):
        display = binary_classification_display

        assert_equal(display.precision, {1: [(0.5, 1, 1), (0.5, 0, 1)]})
        assert_equal(display.recall, {1: [(1, 1, 0), (1, 0, 0)]})
        assert_equal(display.average_precision, {1: [1, 0.5]})
        assert display.estimator_names == ["BC-E1", "BC-E2"]
        assert display.ml_task == "binary-classification"
        assert display.pos_label == 1
        assert display.data_source == "test"

    def test_from_predictions_multiclass_classification(
        self, multiclass_classification_display
    ):
        display = multiclass_classification_display

        assert_equal(
            display.precision,
            {
                0: [(1 / 3, 1, 1), (1 / 3, 0, 0, 1)],
                1: [(1 / 3, 1, 1), (1 / 3, 0, 0, 1)],
                2: [(1 / 3, 1, 1), (1 / 3, 0, 0, 1)],
            },
        )
        assert_equal(
            display.recall,
            {
                0: [(1, 1, 0), (1, 0, 0, 0)],
                1: [(1, 1, 0), (1, 0, 0, 0)],
                2: [(1, 1, 0), (1, 0, 0, 0)],
            },
        )
        assert_equal(
            display.average_precision,
            {0: (1, 1 / 3), 1: (1, 1 / 3), 2: (1, 1 / 3)},
        )
        assert display.estimator_names == ["MC-E1", "MC-E2"]
        assert display.ml_task == "multiclass-classification"
        assert display.pos_label is None
        assert display.data_source == "test"

    def test_plot_binary_classification(self, tmp_path, binary_classification_display):
        display = binary_classification_display
        display.plot()

        # Test `lines_` attribute
        assert isinstance(display.lines_, list)
        assert len(display.lines_) == 2
        assert all(isinstance(line, Line2D) for line in display.lines_)
        assert_equal(display.lines_[0].get_xdata(), (1.0, 1.0, 0.0))
        assert_equal(display.lines_[1].get_xdata(), (1.0, 0.0, 0.0))
        assert_equal(display.lines_[0].get_ydata(), (0.5, 1.0, 1.0))
        assert_equal(display.lines_[1].get_ydata(), (0.5, 0.0, 1.0))
        assert display.lines_[0].get_label() == "BC-E1 #1 (AP = 1.00)"
        assert display.lines_[1].get_label() == "BC-E2 #2 (AP = 0.50)"
        assert display.lines_[0].get_color() != display.lines_[1].get_color()

        # Test `ax_` attribute, its lines, legend and title
        assert isinstance(display.ax_, Axes)
        assert display.ax_.lines[:2] == display.lines_
        assert (
            display.ax_.get_legend().get_title().get_text()
            == "Binary-Classification on $\\bf{test}$ set"
        )
        assert display.ax_.get_xlabel() == "Recall\n(Positive label: 1)"
        assert display.ax_.get_ylabel() == "Precision\n(Positive label: 1)"
        assert display.ax_.get_adjustable() == "box"
        assert display.ax_.get_aspect() in ("equal", 1.0)
        assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)

        # Test `figure_` attribute
        assert display.ax_.figure == display.figure_

    def test_plot_multiclass_classification(
        self, tmp_path, multiclass_classification_display
    ):
        display = multiclass_classification_display
        display.plot()

        # Test `lines_` attribute
        assert isinstance(display.lines_, list)
        assert len(display.lines_) == 6
        assert all(isinstance(line, Line2D) for line in display.lines_)
        assert_equal(display.lines_[0].get_xdata(), (1.0, 1.0, 0.0))
        assert_equal(display.lines_[1].get_xdata(), (1.0, 1.0, 0.0))
        assert_equal(display.lines_[2].get_xdata(), (1.0, 1.0, 0.0))
        assert_equal(display.lines_[3].get_xdata(), (1.0, 0.0, 0.0, 0.0))
        assert_equal(display.lines_[4].get_xdata(), (1.0, 0.0, 0.0, 0.0))
        assert_equal(display.lines_[5].get_xdata(), (1.0, 0.0, 0.0, 0.0))
        assert_equal(display.lines_[0].get_ydata(), (1 / 3, 1.0, 1.0))
        assert_equal(display.lines_[1].get_ydata(), (1 / 3, 1.0, 1.0))
        assert_equal(display.lines_[2].get_ydata(), (1 / 3, 1.0, 1.0))
        assert_equal(display.lines_[3].get_ydata(), (1 / 3, 0.0, 0.0, 1.0))
        assert_equal(display.lines_[4].get_ydata(), (1 / 3, 0.0, 0.0, 1.0))
        assert_equal(display.lines_[5].get_ydata(), (1 / 3, 0.0, 0.0, 1.0))
        assert display.lines_[0].get_label() == "MC-E1 #1 - class 0 (AP = 0.67)"
        assert display.lines_[1].get_label() == "MC-E1 #1 - class 1 (AP = 0.67)"
        assert display.lines_[2].get_label() == "MC-E1 #1 - class 2 (AP = 0.67)"
        assert display.lines_[3].get_label() == "MC-E2 #2 - class 0 (AP = 0.67)"
        assert display.lines_[4].get_label() == "MC-E2 #2 - class 1 (AP = 0.67)"
        assert display.lines_[5].get_label() == "MC-E2 #2 - class 2 (AP = 0.67)"
        assert display.lines_[0].get_color() != display.lines_[3].get_color()
        assert (
            display.lines_[0].get_color()
            == display.lines_[1].get_color()
            == display.lines_[2].get_color()
        )
        assert (
            display.lines_[3].get_color()
            == display.lines_[4].get_color()
            == display.lines_[5].get_color()
        )
        assert display.lines_[0].get_linestyle() != display.lines_[1].get_linestyle()
        assert display.lines_[1].get_linestyle() != display.lines_[2].get_linestyle()
        assert display.lines_[0].get_linestyle() == display.lines_[3].get_linestyle()
        assert display.lines_[1].get_linestyle() == display.lines_[4].get_linestyle()
        assert display.lines_[2].get_linestyle() == display.lines_[5].get_linestyle()

        # Test `ax_` attribute, its lines, legend and title
        assert isinstance(display.ax_, Axes)
        assert display.ax_.lines[:6] == display.lines_
        assert (
            display.ax_.get_legend().get_title().get_text()
            == "Multiclass-Classification on $\\bf{test}$ set"
        )
        assert display.ax_.get_xlabel() == "Recall"
        assert display.ax_.get_ylabel() == "Precision"
        assert display.ax_.get_adjustable() == "box"
        assert display.ax_.get_aspect() in ("equal", 1.0)
        assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)

        # Test `figure_` attribute
        assert display.ax_.figure == display.figure_

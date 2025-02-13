from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images
from numpy import array
from numpy.testing import assert_equal
from skore.sklearn._comparison.precision_recall_curve_display import (
    PrecisionRecallCurveDisplay,
)

CWD = Path(__file__).parent.resolve()


class FakeEstimator:
    def __init__(self, classes: list[int]):
        self.classes_ = classes


class TestPrecisionRecallCurveDisplay:
    def test_from_predictions_binary_classification(self):
        classes = (0, 1)
        y_true = (array((0, 1)), array((0, 1)))
        y_pred = (array((0.2, 0.8)), array((0.8, 0.2)))
        estimators = (FakeEstimator(classes), FakeEstimator(classes))
        display = PrecisionRecallCurveDisplay._from_predictions(
            y_true=y_true,
            y_pred=y_pred,
            estimators=estimators,
            estimator_names=["BC-E1", "BC-E2"],
            ml_task="binary-classification",
            data_source="test",
        )

        assert_equal(display.precision, {1: [(0.5, 1, 1), (0.5, 0, 1)]})
        assert_equal(display.recall, {1: [(1, 1, 0), (1, 0, 0)]})
        assert_equal(display.average_precision, {1: [1, 0.5]})
        assert_equal(display.estimator_names, ["BC-E1", "BC-E2"])
        assert_equal(display.ml_task, "binary-classification")
        assert_equal(display.pos_label, 1)
        assert_equal(display.data_source, "test")

    def test_from_predictions_multiclass_classification(self):
        classes = (0, 1, 2)
        y_true = (array((0, 1, 2)), array((0, 1, 2)))
        y_pred = (
            array(((0.8, 0.2, 0.0), (0.0, 0.8, 0.2), (0.2, 0.0, 0.8))),
            array(((0.0, 0.2, 0.8), (0.8, 0.0, 0.2), (0.2, 0.8, 0.0))),
        )
        estimators = (FakeEstimator(classes), FakeEstimator(classes))
        display = PrecisionRecallCurveDisplay._from_predictions(
            y_true=y_true,
            y_pred=y_pred,
            estimators=estimators,
            estimator_names=["MC-E1", "MC-E2"],
            ml_task="multiclass-classification",
            data_source="test",
        )

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
        assert_equal(display.estimator_names, ["MC-E1", "MC-E2"])
        assert_equal(display.ml_task, "multiclass-classification")
        assert_equal(display.pos_label, None)
        assert_equal(display.data_source, "test")

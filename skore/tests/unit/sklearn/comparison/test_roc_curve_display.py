from numpy import array
from numpy.testing import assert_equal
from skore.sklearn._comparison.roc_curve_display import RocCurveDisplay


class FakeEstimator:
    def __init__(self, classes: list[int]):
        self.classes_ = classes


class TestRocCurveDisplay:
    def test_from_prediction_binary_classification(self):
        classes = (0, 1)
        y_true = (array((0, 1)), array((0, 1)))
        y_pred = (array((0.2, 0.8)), array((0.8, 0.2)))
        estimators = (FakeEstimator(classes), FakeEstimator(classes))
        display = RocCurveDisplay._from_predictions(
            y_true=y_true,
            y_pred=y_pred,
            estimators=estimators,
            estimator_names=["BC-E1", "BC-E2"],
            ml_task="binary-classification",
            data_source="test",
        )

        assert_equal(display.fpr, {1: [(0, 0, 1), (0, 1, 1)]})
        assert_equal(display.tpr, {1: [(0, 1, 1), (0, 0, 1)]})
        assert_equal(display.roc_auc, {1: [1, 0]})
        assert_equal(display.estimator_names, ["BC-E1", "BC-E2"])
        assert_equal(display.ml_task, "binary-classification")
        assert_equal(display.pos_label, 1)
        assert_equal(display.data_source, "test")

    def test_from_prediction_multiclass_classification(self):
        classes = (0, 1, 2)
        y_true = (array((0, 1, 2)), array((0, 1, 2)))
        y_pred = (
            array(((0.8, 0.2, 0.0), (0.0, 0.8, 0.2), (0.2, 0.0, 0.8))),
            array(((0.0, 0.2, 0.8), (0.8, 0.0, 0.2), (0.2, 0.8, 0.0))),
        )
        estimators = (FakeEstimator(classes), FakeEstimator(classes))
        display = RocCurveDisplay._from_predictions(
            y_true=y_true,
            y_pred=y_pred,
            estimators=estimators,
            estimator_names=["E1", "E2"],
            ml_task="multiclass-classification",
            data_source="test",
        )

        assert_equal(
            display.fpr,
            {
                0: [(0.0, 0.0, 1.0), (0.0, 0.5, 1.0, 1.0)],
                1: [(0.0, 0.0, 1.0), (0.0, 0.5, 1.0, 1.0)],
                2: [(0.0, 0.0, 1.0), (0.0, 0.5, 1.0, 1.0)],
            },
        )
        assert_equal(
            display.tpr,
            {
                0: [(0.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0)],
                1: [(0.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0)],
                2: [(0.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0)],
            },
        )
        assert_equal(display.roc_auc, {0: [1.0, 0.0], 1: [1.0, 0.0], 2: [1.0, 0.0]})
        assert_equal(display.estimator_names, ["E1", "E2"])
        assert_equal(display.ml_task, "multiclass-classification")
        assert_equal(display.pos_label, None)
        assert_equal(display.data_source, "test")

    def test_plot_binary_classification(self): ...
    def test_plot_multiclass_classification(self): ...

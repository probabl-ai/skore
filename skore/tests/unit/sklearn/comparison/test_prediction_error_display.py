from pathlib import Path

import pytest
from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images
from numpy import array
from numpy.testing import assert_equal
from skore.sklearn._comparison.prediction_error_display import PredictionErrorDisplay

CWD = Path(__file__).parent.resolve()


@pytest.fixture
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


class TestPrecisionRecallCurveDisplay:
    def test_from_predictions(self, regression_display):
        display = regression_display

        assert_equal(display.y_true, [(-1, 0, 1), (-1, 0, 1)])
        assert_equal(display.y_pred, [(0, 0.2, 0.8), (0.8, 0.2, 0)])
        assert display.estimator_names == ["R-E1", "R-E2"]
        assert display.data_source == "test"

    @pytest.mark.parametrize("kind", ("actual_vs_predicted", "residual_vs_predicted"))
    def test_plot(self, tmp_path, regression_display, kind):
        display = regression_display

        display.plot(kind=kind)
        plt.gcf().savefig(tmp_path / f"pe-{kind}.png")

        assert (
            compare_images(tmp_path / f"pe-{kind}.png", CWD / f"pe-{kind}.png", 0)
            is None
        )

from dataclasses import dataclass

import numpy
import plotly.graph_objects
import pytest
from sklearn.model_selection import StratifiedKFold
from skore.item.cross_validation_item import (
    CrossValidationItem,
    ItemTypeError,
    _hash_numpy,
    _metric_favorability,
)
from skore.sklearn.cross_validation import CrossValidationReporter
from skore.sklearn.cross_validation.cross_validation_reporter import (
    CrossValidationPlots,
)


class FakeEstimator:
    def get_params(self):
        return {"alpha": 3}


class FakeEstimatorNoGetParams:
    pass


@dataclass
class FakeCrossValidationReporter(CrossValidationReporter):
    _cv_results = {
        "test_score": numpy.array([1, 2, 3]),
        "estimator": [FakeEstimator(), FakeEstimator(), FakeEstimator()],
        "fit_time": [1, 2, 3],
    }
    estimator = FakeEstimator()
    X = numpy.array([[1.0]])
    y = numpy.array([1])
    plots = CrossValidationPlots(
        scores=plotly.graph_objects.Figure(),
        timing=plotly.graph_objects.Figure(),
    )
    cv = StratifiedKFold(n_splits=5)


@dataclass
class FakeCrossValidationReporterNoGetParams(CrossValidationReporter):
    _cv_results = {
        "test_score": numpy.array([1, 2, 3]),
        "estimator": [
            FakeEstimatorNoGetParams(),
            FakeEstimatorNoGetParams(),
            FakeEstimatorNoGetParams(),
        ],
        "fit_time": [1, 2, 3],
    }
    estimator = FakeEstimatorNoGetParams()
    X = numpy.array([[1.0]])
    y = numpy.array([1])
    plots = CrossValidationPlots(
        scores=plotly.graph_objects.Figure(),
        timing=plotly.graph_objects.Figure(),
    )
    cv = StratifiedKFold(n_splits=5)


class TestCrossValidationItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.item.item.datetime", MockDatetime)

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            CrossValidationItem.factory(None)

    @pytest.mark.parametrize(
        "reporter",
        [
            pytest.param(FakeCrossValidationReporter(), id="cv_reporter"),
            pytest.param(
                FakeCrossValidationReporterNoGetParams(), id="cv_reporter_no_get_params"
            ),
        ],
    )
    def test_factory(self, mock_nowstr, reporter):
        item = CrossValidationItem.factory(reporter)

        assert item.cv_results_serialized == {"test_score": [1, 2, 3]}
        assert item.estimator_info == {
            "name": reporter.estimator.__class__.__name__,
            "params": (
                {}
                if isinstance(reporter.estimator, FakeEstimatorNoGetParams)
                else {"alpha": {"value": "3", "default": True}}
            ),
            "module": "tests.unit.item.test_cross_validation_item",
        }
        assert item.X_info == {
            "nb_cols": 1,
            "nb_rows": 1,
            "hash": _hash_numpy(FakeCrossValidationReporter.X),
        }
        assert item.y_info == {"hash": _hash_numpy(FakeCrossValidationReporter.y)}
        assert item.cv_info == {
            "n_splits": "5",
            "random_state": "None",
            "shuffle": "False",
        }
        assert isinstance(item.plots_bytes, dict)
        assert isinstance(item.plots, dict)
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    def test_get_serializable_dict(self, monkeypatch, mock_nowstr):
        monkeypatch.setattr(
            "skore.item.cross_validation_item.CrossValidationReporter",
            FakeCrossValidationReporter,
        )

        reporter = FakeCrossValidationReporter()
        item = CrossValidationItem.factory(reporter)
        serializable = item.as_serializable_dict()

        assert serializable["updated_at"] == mock_nowstr
        assert serializable["created_at"] == mock_nowstr
        assert serializable["value"]["scalar_results"] == [
            {
                "name": "Mean test score",
                "value": 2,
                "stddev": 1.0,
                "favorability": "greater_is_better",
            }
        ]
        assert serializable["value"]["tabular_results"] == [
            {
                "name": "Cross validation results",
                "columns": ["test_score"],
                "data": [(1,), (2,), (3,)],
                "favorability": [
                    "greater_is_better",
                ],
            }
        ]

    @pytest.mark.parametrize(
        "metric,expected",
        [
            # greater_is_better metrics (exact matches)
            ("accuracy", "greater_is_better"),
            ("balanced_accuracy", "greater_is_better"),
            ("top_k_accuracy", "greater_is_better"),
            ("average_precision", "greater_is_better"),
            ("f1", "greater_is_better"),
            ("precision", "greater_is_better"),
            ("recall", "greater_is_better"),
            ("jaccard", "greater_is_better"),
            ("roc_auc", "greater_is_better"),
            ("r2", "greater_is_better"),
            # greater_is_better metrics (pattern matches)
            ("weighted_f1", "greater_is_better"),
            ("macro_precision", "greater_is_better"),
            ("micro_recall", "greater_is_better"),
            # greater_is_better by convention (_score suffix)
            ("custom_score", "greater_is_better"),
            ("validation_score", "greater_is_better"),
            # greater_is_better by convention (neg_ prefix)
            ("neg_mean_squared_error", "greater_is_better"),
            ("neg_log_loss", "greater_is_better"),
            # the same one but without the neg_ prefix
            ("mean_squared_error", "lower_is_better"),
            ("log_loss", "lower_is_better"),
            # lower_is_better metrics (exact matches)
            ("fit_time", "lower_is_better"),
            ("score_time", "lower_is_better"),
            # lower_is_better by convention (suffixes)
            ("mean_squared_error", "lower_is_better"),
            ("mean_absolute_error", "lower_is_better"),
            ("binary_crossentropy_loss", "lower_is_better"),
            ("hinge_loss", "lower_is_better"),
            ("entropy_deviance", "lower_is_better"),
            # unknown metrics
            ("custom_metric", "unknown"),
            ("undefined", "unknown"),
            ("", "unknown"),
        ],
    )
    def test_metric_favorability(self, metric, expected):
        """Test the _metric_favorability function with various metric names.

        Non-regression test for:
        https://github.com/probabl-ai/skore/issues/1061
        """
        assert _metric_favorability(metric) == expected

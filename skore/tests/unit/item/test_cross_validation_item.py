from dataclasses import dataclass

import numpy
import plotly.graph_objects
import pytest
from sklearn.model_selection import StratifiedKFold
from skore.item.cross_validation_item import (
    CrossValidationItem,
    ItemTypeError,
    _hash_numpy,
)


class FakeEstimator:
    def get_params(self):
        return {}


@dataclass
class FakeCrossValidationReporter:
    _cv_results = {
        "test_score": numpy.array([1, 2, 3]),
        "estimator": [FakeEstimator(), FakeEstimator(), FakeEstimator()],
        "fit_time": [1, 2, 3],
    }
    estimator = FakeEstimator()
    X = numpy.array([[1.0]])
    y = numpy.array([1])
    plot = plotly.graph_objects.Figure()
    cv = StratifiedKFold(n_splits=5)


class TestCrossValidationItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.item.item.datetime", MockDatetime)

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            CrossValidationItem.factory(None)

    def test_factory(self, monkeypatch, mock_nowstr):
        monkeypatch.setattr(
            "skore.item.cross_validation_item.CrossValidationReporter",
            FakeCrossValidationReporter,
        )

        reporter = FakeCrossValidationReporter()
        item = CrossValidationItem.factory(reporter)

        assert item.cv_results_serialized == {"test_score": [1, 2, 3]}
        assert item.estimator_info == {
            "name": "FakeEstimator",
            "params": {},
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
        assert isinstance(item.plot_bytes, bytes)
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    def test_get_serializable_dict(self, monkeypatch, mock_nowstr):
        monkeypatch.setattr(
            "skore.item.cross_validation_item.CrossValidationReporter",
            FakeCrossValidationReporter,
        )

        reporter = FakeCrossValidationReporter()
        item = CrossValidationItem.factory(reporter)
        serializable = item.get_serializable_dict()

        assert serializable["updated_at"] == mock_nowstr
        assert serializable["created_at"] == mock_nowstr
        assert (
            serializable["media_type"] == "application/vnd.skore.cross_validation+json"
        )
        assert serializable["value"] is not None

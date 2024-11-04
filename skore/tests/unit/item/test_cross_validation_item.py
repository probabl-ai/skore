import numpy
import pytest
from skore.item import CrossValidationItem, ItemTypeError


class TestCrossValidationItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.item.item.datetime", MockDatetime)

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            CrossValidationItem.factory(
                cv_results=None,
                estimator=None,
                X=None,
                y=None,
            )

    def test_factory(self, monkeypatch, mock_nowstr):
        monkeypatch.setattr(
            "skore.item.cross_validation_item._hash_numpy", lambda x: ""
        )

        class MyEstimator:
            def get_params(self):
                return {}

        item = CrossValidationItem.factory(
            cv_results={
                "test_score": numpy.array([1, 2, 3]),
                "estimator": [MyEstimator(), MyEstimator(), MyEstimator()],
                "fit_time": [1, 2, 3],
            },
            cv_results_items_history=[],
            estimator=MyEstimator(),
            X=[[1.0]],
            y=[1],
        )

        assert item.cv_results_serialized == {"test_score": [1, 2, 3]}
        assert item.estimator_info == {"name": "MyEstimator", "params": "{}"}
        assert item.X_info == {"nb_cols": 1, "nb_rows": 1, "hash": ""}
        assert item.y_info == {"hash": ""}
        assert isinstance(item.plot_bytes, bytes)
        assert isinstance(item.aggregation_plot_bytes, bytes)
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

import numpy as np
import polars as pl
import pytest
from sklearn.dummy import DummyRegressor

from skore import evaluate


@pytest.fixture
def polars_data():
    rng = np.random.default_rng(0)
    X = pl.DataFrame(rng.normal(size=(40, 3)))
    y = pl.Series(rng.normal(size=40))
    return X, y


def test_evaluate_metrics_with_polars(polars_data):
    X, y = polars_data
    report = evaluate(DummyRegressor(), X=X, y=y, splitter=0.2)
    assert report.metrics.summarize() is not None


def test_checks_with_polars(polars_data):
    X, y = polars_data
    report = evaluate(DummyRegressor(), X=X, y=y, splitter=0.2)
    summary = report.checks.summarize()
    assert summary is not None


def test_data_summarize_with_polars(polars_data):
    X, y = polars_data
    report = evaluate(DummyRegressor(), X=X, y=y, splitter=0.2)
    display = report.data.summarize()
    assert display.frame(kind="dataset") is not None

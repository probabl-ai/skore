import pandas as pd
import polars as pl
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

from skore import CrossValidationReport, EstimatorReport


def _columns(report, **kwargs):
    display = report.data.summarize(with_plots=False, **kwargs)
    return list(display.summary["dataframe"].columns)


def test_colliding_target_name_uses_stable_suffix():
    """A feature named like the target is not suffixed with ``_right``."""
    X, y = make_regression(n_samples=40, n_features=4, random_state=0)
    features = pd.DataFrame(X, columns=["target", "f1", "f2", "f3"])
    target = pd.Series(y, name="target")
    report = EstimatorReport(
        LinearRegression(),
        X_train=features,
        y_train=target,
        X_test=features,
        y_test=target,
    )

    both_columns = _columns(report)
    test_columns = _columns(report, data_source="test")
    assert both_columns == ["target", "f1", "f2", "f3", "target__y"]
    assert test_columns == ["target", "f1", "f2", "f3", "target__y"]
    assert "target_right" not in both_columns
    assert "target_right" not in test_columns

    cv_report = CrossValidationReport(LinearRegression(), features, target, splitter=2)
    cv_columns = _columns(cv_report)
    assert cv_columns == ["target", "f1", "f2", "f3", "target__y"]
    assert "target_right" not in cv_columns


def test_colliding_polars_target_name_uses_stable_suffix():
    """The same suffix is used when both frames are Polars."""
    X, y = make_regression(n_samples=40, n_features=4, random_state=0)
    features = pl.DataFrame(
        {"target": X[:, 0], "f1": X[:, 1], "f2": X[:, 2], "f3": X[:, 3]}
    )
    target = pl.Series("target", y)
    report = EstimatorReport(
        LinearRegression(),
        X_train=features,
        y_train=target,
        X_test=features,
        y_test=target,
    )
    assert _columns(report, data_source="test") == [
        "target",
        "f1",
        "f2",
        "f3",
        "target__y",
    ]

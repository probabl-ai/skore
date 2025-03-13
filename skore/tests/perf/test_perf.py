import pytest
import xxhash
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from skore import EstimatorReport, config_context, set_config

set_config(show_progress=False)


def test_manual():
    X, y = make_regression(n_samples=10_000, n_features=1_00, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report.metrics.report_metrics(data_source="X_y", X=X, y=y)


@pytest.mark.parametrize(
    "hash_func",
    [
        "md5",
        pytest.param(xxhash.xxh32(), id="xxhash32"),
        pytest.param(xxhash.xxh3_64(), id="xxhash3_64"),
    ],
)
def test_hash(hash_func, benchmark):
    X, y = make_regression(n_samples=10_000, n_features=100, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    with config_context(hash_func=hash_func):
        benchmark.pedantic(
            report.metrics.report_metrics,
            kwargs=dict(data_source="X_y", X=X, y=y),
            warmup_rounds=1,
            rounds=10,
            iterations=20,
        )

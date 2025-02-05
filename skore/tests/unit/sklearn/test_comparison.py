import warnings
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from skore import ComparisonReport, EstimatorReport


@pytest.fixture
def logistic_regression_report():
    X, y = make_classification(n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    estimator = LogisticRegression()

    return EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@pytest.fixture
def linear_regression_report():
    X, y = make_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    estimator = LinearRegression()

    return EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def test_comparison_report(logistic_regression_report):
    comp = ComparisonReport([logistic_regression_report, logistic_regression_report])

    comp.metrics.help()
    comp.metrics.report_metrics()
    comp.metrics.brier_score()
    comp.metrics.plot.roc()
    print(comp.metrics.accuracy())
    print(comp.metrics.accuracy(aggregate="mean"))
    print(comp.metrics.accuracy(aggregate=["mean", "std"]))


def test_comparison_report_different_estimators():
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    logistic_regression = LogisticRegression().fit(X_train, y_train)
    dummy_classifier = DummyClassifier().fit(X_train, y_train)

    from skore import ComparisonReport

    comp = ComparisonReport(
        [
            EstimatorReport(logistic_regression, X_test=X_test, y_test=y_test),
            EstimatorReport(dummy_classifier, X_test=X_test, y_test=y_test),
        ]
    )

    comp.metrics.help()
    with warnings.catch_warnings(action="ignore"):
        comp.metrics.report_metrics()
    comp.metrics.brier_score()
    comp.metrics.plot.roc()
    print(comp.metrics.accuracy())
    print(comp.metrics.accuracy(aggregate="mean"))
    print(comp.metrics.accuracy(aggregate=["mean", "std"]))


def test_example(tmp_path):
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from skore import ComparisonReport, EstimatorReport, Project, train_test_split

    X, y = make_classification(n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    log_reg = LogisticRegression()
    est_report_lr = EstimatorReport(
        log_reg, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    rf = RandomForestClassifier(max_depth=2, random_state=0)
    est_report_rf = EstimatorReport(
        rf, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    project = Project(tmp_path / "my-project.skore")
    project.put("est_rep", est_report_lr)
    project.put("est_rep", est_report_rf)

    comp1 = ComparisonReport([est_report_lr, est_report_rf])
    comp2 = ComparisonReport(project.get("est_rep", version="all"))

    print(comp1.metrics.accuracy())
    print(comp2.metrics.accuracy())


def test_no_train_data():
    """If EstimatorReports do not contain training data, computing metrics
    with `data_source="train"` should fail."""
    X, y = make_classification(n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    est_report = EstimatorReport(log_reg, X_test=X_test, y_test=y_test)

    comp = ComparisonReport([est_report, est_report])

    with pytest.raises(ValueError, match="No train data"):
        comp.metrics.accuracy(data_source="train")


def test_comparison_report_init_len(logistic_regression_report):
    """If we have less than 2 estimator reports to compare, the comparison report raises
    an exception."""

    with pytest.raises(
        ValueError, match="At least 2 instances of EstimatorReport are needed"
    ):
        ComparisonReport([logistic_regression_report])


def test_comparison_report_init_wrong_type(logistic_regression_report):
    """If the input is not a list of `EstimatorReport`s, raise."""

    with pytest.raises(
        TypeError, match="object of type 'EstimatorReport' has no len()"
    ):
        ComparisonReport(logistic_regression_report)

    with pytest.raises(
        TypeError, match="Only instances of EstimatorReport are allowed"
    ):
        ComparisonReport([logistic_regression_report, None])

    with pytest.raises(
        TypeError, match="Only instances of EstimatorReport are allowed"
    ):
        ComparisonReport([None, logistic_regression_report])


def test_comparison_report_init_deepcopy(logistic_regression_report):
    """If an estimator report is modified outside of the comparator, it is not modified
    inside the comparator."""

    comp = ComparisonReport([logistic_regression_report, logistic_regression_report])

    # check if the deepcopy work well
    assert comp.estimator_reports_[0]._hash == logistic_regression_report._hash

    # modify the estimator report outside of the comparator
    my_new_hash_value = 0
    logistic_regression_report._hash = my_new_hash_value

    # check if there is no impact on the estimator report in the comparator (its value
    # has not changed)
    assert comp.estimator_reports_[0]._hash != my_new_hash_value


def test_comparison_report_init_MissingTrainingDataWarning(capsys):
    """Raise a warning if there is no training data (`None`) for any estimator
    report."""

    X, y = make_classification(n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    estimator = LogisticRegression()
    estimator.fit(X_train, y_train)

    est_report = EstimatorReport(
        estimator,
        X_test=X_test,
        y_test=y_test,
    )

    ComparisonReport([est_report, est_report])

    captured = capsys.readouterr()

    assert "MissingTrainingDataWarning" in captured.out


def test_comparison_report_init_MissingTestDataWarning(capsys):
    """Raise a warning if there is no test data (`None`) for any estimator
    report."""

    X, y = make_classification(n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    estimator = LogisticRegression()
    estimator.fit(X_train, y_train)

    est_report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
    )

    ComparisonReport([est_report, est_report])

    captured = capsys.readouterr()

    assert "MissingTestDataWarning" in captured.out


def test_comparison_report_init_different_ml_usecases(
    linear_regression_report,
    logistic_regression_report,
):
    with pytest.raises(
        ValueError, match="Not all estimators are in the same ML usecase"
    ):
        ComparisonReport([linear_regression_report, logistic_regression_report])


def test_comparison_report_init_different_training_data():
    X, y = make_classification(n_classes=2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    report_1 = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    # Different random state, so the train and test data are different
    X_train, _, y_train, _ = train_test_split(X, y, random_state=2)

    report_2 = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    with pytest.raises(
        ValueError, match="Not all estimators have the same training data"
    ):
        ComparisonReport([report_1, report_2])


def test_comparison_report_init_different_test_data():
    X, y = make_classification(n_classes=2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    report_1 = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    # Different random state, so the train and test data are different
    _, X_test, _, y_test = train_test_split(X, y, random_state=2)

    report_2 = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    with pytest.raises(
        ValueError, match="Not all estimators have the same testing data"
    ):
        ComparisonReport([report_1, report_2])


def test_comparison_report_help(capsys, logistic_regression_report):
    """Check that the help method writes to the console."""
    report = ComparisonReport([logistic_regression_report, logistic_regression_report])
    report.help()

    assert "Tools to compare estimators" in capsys.readouterr().out


def test_comparison_report_repr(logistic_regression_report):
    """Check that __repr__ returns a string starting with the expected prefix."""
    report = ComparisonReport([logistic_regression_report, logistic_regression_report])
    repr_str = repr(report)

    assert "skore.ComparisonReport" in repr_str
    assert "help()" in repr_str


def test_comparison_report_pickle(tmp_path, logistic_regression_report):
    """Check that we can pickle a comparison report."""
    with BytesIO() as f:
        joblib.dump(
            ComparisonReport([logistic_regression_report, logistic_regression_report]),
            f,
        )


@pytest.mark.parametrize(
    "metric_name,metric_value",
    [
        ("accuracy", 1),
        ("precision", 2),
        ("recall", 3),
        ("brier_score", 4),
        ("roc_auc", 5),
        ("log_loss", 6),
    ],
)
def test_estimator_report_metrics_binary_classification(
    monkeypatch,
    metric_name,
    metric_value,
):
    def dummy(*args, **kwargs):
        return pd.DataFrame([[metric_value]])

    monkeypatch.setattr(
        f"skore.sklearn._estimator.metrics_accessor._MetricsAccessor.{metric_name}",
        dummy,
    )

    X, y = make_classification(n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    logistic_regression_report = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    comp = ComparisonReport([logistic_regression_report, logistic_regression_report])

    # ensure metric is valid
    result = getattr(comp.metrics, metric_name)()
    np.testing.assert_array_equal(result.to_numpy(), [[metric_value], [metric_value]])

    # ensure metric is valid even from the cache
    result = getattr(comp.metrics, metric_name)()
    np.testing.assert_array_equal(result.to_numpy(), [[metric_value], [metric_value]])


@pytest.mark.parametrize(
    "metric_name,metric_value",
    [
        ("rmse", 1),
        ("r2", 2),
    ],
)
def test_estimator_report_metrics_regression(
    monkeypatch,
    metric_name,
    metric_value,
):
    def dummy(*args, **kwargs):
        return pd.DataFrame([[metric_value]])

    monkeypatch.setattr(
        f"skore.sklearn._estimator.metrics_accessor._MetricsAccessor.{metric_name}",
        dummy,
    )

    X, y = make_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    estimator_report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    comp = ComparisonReport([estimator_report, estimator_report])

    # ensure metric is valid
    result = getattr(comp.metrics, metric_name)()
    np.testing.assert_array_equal(result.to_numpy(), [[metric_value], [metric_value]])

    # ensure metric is valid even from the cache
    result = getattr(comp.metrics, metric_name)()
    np.testing.assert_array_equal(result.to_numpy(), [[metric_value], [metric_value]])

import re

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skore import Display, EstimatorReport


@pytest.fixture
def binary_classification_data():
    """Create a binary classification dataset and return fitted estimator and data."""
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return RandomForestClassifier().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def binary_classification_data_svc():
    """Create a binary classification dataset and return fitted estimator and data.
    The estimator is a SVC that does not support `predict_proba`.
    """
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return SVC().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def multiclass_classification_data():
    """Create a multiclass classification dataset and return fitted estimator and
    data."""
    X, y = make_classification(
        n_classes=3, n_clusters_per_class=1, random_state=42, n_informative=10
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return RandomForestClassifier().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def multiclass_classification_data_svc():
    """Create a multiclass classification dataset and return fitted estimator and
    data. The estimator is a SVC that does not support `predict_proba`.
    """
    X, y = make_classification(
        n_classes=3, n_clusters_per_class=1, random_state=42, n_informative=10
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return SVC().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def binary_classification_data_pipeline():
    """Create a binary classification dataset and return fitted pipeline and data."""
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    estimator = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    return estimator.fit(X_train, y_train), X_test, y_test


@pytest.fixture
def regression_data():
    """Create a regression dataset and return fitted estimator and data."""
    X, y = make_regression(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return LinearRegression().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def regression_multioutput_data():
    """Create a regression dataset and return fitted estimator and data."""
    X, y = make_regression(n_targets=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return LinearRegression().fit(X_train, y_train), X_test, y_test


def test_summary_display_frame_flat_index(binary_classification_data):
    """Check the behaviour of `flat_index` in `summarize`."""
    estimator, X_test, y_test = binary_classification_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    result = report.metrics.summarize().frame(flat_index=True)
    assert result.shape == (8, 1)
    assert isinstance(result.index, pd.Index)
    assert result.index.tolist() == [
        "precision_0",
        "precision_1",
        "recall_0",
        "recall_1",
        "roc_auc",
        "brier_score",
        "fit_time",
        "predict_time",
    ]

    assert result.columns.tolist() == ["RandomForestClassifier"]

    result = report.metrics.summarize().frame(flat_index=False)
    assert result.shape == (8, 1)
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.tolist() == [
        ("precision", 0),
        ("precision", 1),
        ("recall", 0),
        ("recall", 1),
        ("roc_auc", ""),
        ("brier_score", ""),
        ("fit_time", ""),
        ("predict_time", ""),
    ]

    assert result.columns.tolist() == ["RandomForestClassifier"]


def _normalize_metric_name(column):
    """Helper to normalize the metric name present in a pandas index that could be
    a multi-index or single-index."""
    # if we have a multi-index, then the metric name is on level 0
    s = column[0] if isinstance(column, tuple) else column
    # Remove spaces and underscores and (s) suffix
    s = s.lower().replace(" (s)", "")
    return re.sub(r"[^a-zA-Z]", "", s)


def _check_results_summarize(result, expected_metrics, expected_nb_stats):
    assert isinstance(result, pd.DataFrame)
    assert len(result.index) == expected_nb_stats

    normalized_expected = {
        _normalize_metric_name(metric) for metric in expected_metrics
    }
    for column in result.index:
        normalized_column = _normalize_metric_name(column)
        matches = [
            metric for metric in normalized_expected if metric == normalized_column
        ]
        assert len(matches) == 1, (
            f"No match found for column '{column}' in expected metrics: "
            f" {expected_metrics}"
        )


@pytest.mark.parametrize("pos_label, nb_stats", [(None, 2), (1, 1)])
@pytest.mark.parametrize("data_source", ["test", "X_y"])
def test_summary_display_frame_binary(
    binary_classification_data,
    binary_classification_data_svc,
    pos_label,
    nb_stats,
    data_source,
):
    """Check the behaviour of the `summarize` method with binary
    classification. We test both with an SVC that does not support `predict_proba` and a
    RandomForestClassifier that does.
    """
    estimator, X_test, y_test = binary_classification_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    kwargs = {"X": X_test, "y": y_test} if data_source == "X_y" else {}
    display = report.metrics.summarize(
        pos_label=pos_label, data_source=data_source, **kwargs
    )
    assert isinstance(display, Display)
    result = display.frame(flat_index=False)
    assert "Favorability" not in result.columns
    expected_metrics = (
        "precision",
        "recall",
        "roc_auc",
        "brier_score",
        "fit_time",
        "predict_time",
    )
    # depending on `pos_label`, we report a stats for each class or not for
    # precision and recall
    expected_nb_stats = 2 * nb_stats + 4
    _check_results_summarize(result, expected_metrics, expected_nb_stats)

    # Repeat the same experiment where we the target labels are not [0, 1] but
    # ["neg", "pos"]. We check that we don't get any error.
    target_names = np.array(["neg", "pos"], dtype=object)
    pos_label_name = target_names[pos_label] if pos_label is not None else pos_label
    y_test = target_names[y_test]
    estimator = clone(estimator).fit(X_test, y_test)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    kwargs = {"X": X_test, "y": y_test} if data_source == "X_y" else {}
    display = report.metrics.summarize(
        pos_label=pos_label_name, data_source=data_source, **kwargs
    )
    assert isinstance(display, Display)
    result = display.frame(flat_index=False)
    expected_metrics = (
        "precision",
        "recall",
        "roc_auc",
        "brier_score",
        "fit_time",
        "predict_time",
    )
    # depending on `pos_label`, we report a stats for each class or not for
    # precision and recall
    expected_nb_stats = 2 * nb_stats + 4
    _check_results_summarize(result, expected_metrics, expected_nb_stats)

    estimator, X_test, y_test = binary_classification_data_svc
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    kwargs = {"X": X_test, "y": y_test} if data_source == "X_y" else {}
    display = report.metrics.summarize(
        pos_label=pos_label, data_source=data_source, **kwargs
    )
    assert isinstance(display, Display)
    result = display.frame(flat_index=False)
    expected_metrics = ("precision", "recall", "roc_auc", "fit_time", "predict_time")
    # depending on `pos_label`, we report a stats for each class or not for
    # precision and recall
    expected_nb_stats = 2 * nb_stats + 3
    _check_results_summarize(result, expected_metrics, expected_nb_stats)


@pytest.mark.parametrize("data_source", ["test", "X_y"])
def test_summary_display_frame_multiclass(
    multiclass_classification_data, multiclass_classification_data_svc, data_source
):
    """Check the behaviour of the `summarize` method with multiclass
    classification.
    """
    estimator, X_test, y_test = multiclass_classification_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    kwargs = {"X": X_test, "y": y_test} if data_source == "X_y" else {}
    display = report.metrics.summarize(data_source=data_source, **kwargs)
    assert isinstance(display, Display)
    result = display.frame(flat_index=False)
    assert "Favorability" not in result.columns
    expected_metrics = (
        "precision",
        "recall",
        "roc_auc",
        "log_loss",
        "fit_time",
        "predict_time",
    )
    # since we are not averaging by default, we report 3 statistics for
    # precision, recall and roc_auc
    expected_nb_stats = 3 * 3 + 3
    _check_results_summarize(result, expected_metrics, expected_nb_stats)

    estimator, X_test, y_test = multiclass_classification_data_svc
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    kwargs = {"X": X_test, "y": y_test} if data_source == "X_y" else {}
    display = report.metrics.summarize(data_source=data_source, **kwargs)
    assert isinstance(display, Display)
    result = display.frame(flat_index=False)
    expected_metrics = ("precision", "recall", "fit_time", "predict_time")
    # since we are not averaging by default, we report 3 statistics for
    # precision and recall
    expected_nb_stats = 3 * 2 + 2
    _check_results_summarize(result, expected_metrics, expected_nb_stats)


@pytest.mark.parametrize("data_source", ["test", "X_y"])
def test_summary_display_frame_regression(regression_data, data_source):
    """Check the behaviour of the `summarize` method with regression."""
    estimator, X_test, y_test = regression_data
    kwargs = {"X": X_test, "y": y_test} if data_source == "X_y" else {}
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize(data_source=data_source, **kwargs)
    assert isinstance(display, Display)
    result = display.frame(flat_index=False)
    assert "Favorability" not in result.columns
    expected_metrics = ("r2", "rmse", "fit_time", "predict_time")
    _check_results_summarize(result, expected_metrics, len(expected_metrics))


def test_estimator_report_summarize_scoring_kwargs(
    regression_multioutput_data, multiclass_classification_data
):
    """Check the behaviour of the `summarize` method with scoring kwargs."""
    estimator, X_test, y_test = regression_multioutput_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    assert hasattr(report.metrics, "summarize")
    display = report.metrics.summarize(scoring_kwargs={"multioutput": "raw_values"})
    assert isinstance(display, Display)
    result = display.frame(flat_index=False)
    assert result.shape == (6, 1)
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["metric", "output"]

    estimator, X_test, y_test = multiclass_classification_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    assert hasattr(report.metrics, "summarize")
    display = report.metrics.summarize(scoring_kwargs={"average": None})
    assert isinstance(display, Display)
    result = display.frame(flat_index=False)
    assert result.shape == (12, 1)
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["metric", "label"]

    display = report.metrics.summarize(scoring_kwargs={"average": "macro"})
    assert isinstance(display, Display)
    result = display.frame(flat_index=False)
    assert result.shape == (6, 1)
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["metric", "average"]


def test_summary_display_frame_scoring_names(binary_classification_data):
    """Check the behaviour of `scoring_names` with the `frame` method."""
    estimator, X_test, y_test = binary_classification_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    assert hasattr(report.metrics, "summarize")
    display = report.metrics.summarize()

    result = display.frame()
    assert result.index.tolist() == [
        "precision_0",
        "precision_1",
        "recall_0",
        "recall_1",
        "roc_auc",
        "brier_score",
        "fit_time",
        "predict_time",
    ]

    result = display.frame(scoring_names="verbose")
    assert result.index.tolist() == [
        "Precision 0",
        "Precision 1",
        "Recall 0",
        "Recall 1",
        "ROC AUC",
        "Brier score",
        "Fit time (s)",
        "Predict time (s)",
    ]
    result = display.frame(scoring_names="verbose", flat_index=False)
    assert result.index.tolist() == [
        ("Precision", 0),
        ("Precision", 1),
        ("Recall", 0),
        ("Recall", 1),
        ("ROC AUC", ""),
        ("Brier score", ""),
        ("Fit time (s)", ""),
        ("Predict time (s)", ""),
    ]

    scoring_names = {"precision": "Prec", "recall": "Rec"}
    result = display.frame(scoring_names=scoring_names)
    assert result.index.tolist() == [
        "Prec 0",
        "Prec 1",
        "Rec 0",
        "Rec 1",
        "roc_auc",
        "brier_score",
        "fit_time",
        "predict_time",
    ]


def test_summary_display_frame_indicator_favorability(
    binary_classification_data,
):
    """Check that the behaviour of `indicator_favorability` is correct."""
    estimator, X_test, y_test = binary_classification_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()
    assert isinstance(display, Display)
    result = display.frame(
        scoring_names="verbose", flat_index=False, indicator_favorability=True
    )
    assert "favorability" in result.columns
    indicator = result["favorability"]
    assert indicator["Precision"].tolist() == ["(↗︎)", "(↗︎)"]
    assert indicator["Recall"].tolist() == ["(↗︎)", "(↗︎)"]
    assert indicator["ROC AUC"].tolist() == ["(↗︎)"]
    assert indicator["Brier score"].tolist() == ["(↘︎)"]

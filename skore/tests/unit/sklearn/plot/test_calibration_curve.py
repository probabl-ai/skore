"""Tests for the calibration curve display."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.gridspec import GridSpec
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from skore import ComparisonReport, EstimatorReport, train_test_split
from skore.sklearn._plot.metrics.calibration_curve import CalibrationCurveDisplay
from skore.sklearn.types import YPlotData


@pytest.fixture
def binary_classification_data():
    """Create binary classification data for testing."""
    X, y = make_classification(
        n_samples=1000, n_classes=2, n_informative=4, random_state=42
    )
    return train_test_split(X=X, y=y, random_state=42, as_dict=True)


@pytest.fixture
def binary_classification_model():
    """Create a classification model for testing."""
    return LogisticRegression(random_state=42)


def test_calibration_curve_display_init():
    """Test that CalibrationCurveDisplay can be initialized with proper attributes."""
    # Create dummy data
    prob_true = {1: [np.array([0.1, 0.3, 0.5, 0.7, 0.9])]}
    prob_pred = {1: [np.array([0.2, 0.4, 0.6, 0.8, 1.0])]}
    y_prob = [np.array([0.2, 0.4, 0.6, 0.8, 1.0])]

    # Initialize the display
    display = CalibrationCurveDisplay(
        prob_true=prob_true,
        prob_pred=prob_pred,
        y_prob=y_prob,
        estimator_names=["TestEstimator"],
        pos_label=1,
        data_source="test",
        ml_task="binary-classification",
        report_type="estimator",
        n_bins=5,
        strategy="uniform",
    )

    # Check that the attributes are set correctly
    assert display.prob_true == prob_true
    assert display.prob_pred == prob_pred
    assert display.y_prob == y_prob
    assert display.estimator_names == ["TestEstimator"]
    assert display.pos_label == 1
    assert display.data_source == "test"
    assert display.ml_task == "binary-classification"
    assert display.report_type == "estimator"
    assert display.n_bins == 5
    assert display.strategy == "uniform"


def test_calibration_curve_from_report(
    binary_classification_data, binary_classification_model
):
    """Test that the calibration curve can be created from an EstimatorReport."""
    # Create a report
    report = EstimatorReport(binary_classification_model, **binary_classification_data)

    # Get the calibration curve display
    display = report.metrics.calibration_curve(pos_label=1)

    # Check that the display is of the right type
    assert isinstance(display, CalibrationCurveDisplay)

    # Check basic attributes
    assert display.pos_label == 1
    assert display.data_source == "test"
    assert display.ml_task == "binary-classification"
    assert display.report_type == "estimator"


def test_calibration_curve_plotting():
    """Test that the calibration curve plotting works correctly."""
    # Create synthetic data
    X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Fit a model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate calibration curve manually
    prob_true, prob_pred = calibration_curve(
        y_test, y_prob, n_bins=5, strategy="uniform"
    )

    # Create YPlotData objects for the display
    y_true_data = [
        YPlotData(estimator_name="LogisticRegression", split_index=None, y=y_test)
    ]
    y_pred_data = [
        YPlotData(
            estimator_name="LogisticRegression",
            split_index=None,
            y=model.predict_proba(X_test),
        )
    ]

    # Create the display directly
    display = CalibrationCurveDisplay._compute_data_for_display(
        y_true=y_true_data,
        y_pred=y_pred_data,
        report_type="estimator",
        estimator_names=["LogisticRegression"],
        ml_task="binary-classification",
        data_source="test",
        pos_label=1,
        n_bins=5,
        strategy="uniform",
    )

    # Check that the computed values match
    np.testing.assert_allclose(display.prob_true[1][0], prob_true)
    np.testing.assert_allclose(display.prob_pred[1][0], prob_pred)

    # Test that the plot method works without error
    fig, (ax, hist_ax) = plt.subplots(nrows=2, figsize=(8, 8), height_ratios=[2, 1])
    display.plot(ax=ax, hist_ax=hist_ax)
    plt.close(fig)


def test_multiple_models_calibration():
    """Test calibration curves with multiple models."""
    # Create data
    X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Create and fit models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Naive Bayes": GaussianNB(),
        "SVC": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
    }

    # Fit all models
    for _name, model in models.items():
        model.fit(X_train, y_train)

    # Create YPlotData objects for each model
    y_true_data = []
    y_pred_data = []

    for name, model in models.items():
        y_true_data.append(YPlotData(estimator_name=name, split_index=None, y=y_test))
        y_pred_proba = model.predict_proba(X_test)
        y_pred_data.append(
            YPlotData(estimator_name=name, split_index=None, y=y_pred_proba)
        )

    # Create the display
    display = CalibrationCurveDisplay._compute_data_for_display(
        y_true=y_true_data,
        y_pred=y_pred_data,
        report_type="comparison-estimator",
        estimator_names=list(models.keys()),
        ml_task="binary-classification",
        data_source="test",
        pos_label=1,
        n_bins=10,
        strategy="uniform",
    )

    # Check that we have data for each model
    assert len(display.prob_true[1]) == len(models)
    assert len(display.prob_pred[1]) == len(models)
    assert len(display.y_prob) == len(models)

    # Create a plot
    fig, (ax, hist_ax) = plt.subplots(nrows=2, figsize=(10, 10), height_ratios=[2, 1])
    display.plot(ax=ax, hist_ax=hist_ax)
    plt.close(fig)


def test_calibration_curve_parameters():
    """Test that the calibration curve parameters are correctly handled."""
    # Create synthetic data and use keyword arguments for train_test_split
    X, y = make_classification(n_samples=1000, random_state=42)
    split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)

    # Create the report
    report = EstimatorReport(LogisticRegression(random_state=42), **split_data)

    # Test default parameters
    display = report.metrics.calibration_curve(pos_label=1)
    assert display.n_bins == 5
    assert display.strategy == "uniform"

    # Test with custom parameters
    display = report.metrics.calibration_curve(
        pos_label=1, n_bins=10, strategy="quantile"
    )
    assert display.n_bins == 10
    assert display.strategy == "quantile"

    # Test invalid strategy
    with pytest.raises(ValueError):
        CalibrationCurveDisplay._compute_data_for_display(
            y_true=[
                YPlotData(estimator_name="test", split_index=None, y=np.array([0, 1]))
            ],
            y_pred=[
                YPlotData(
                    estimator_name="test",
                    split_index=None,
                    y=np.array([[0.1, 0.9], [0.2, 0.8]]),
                )
            ],
            report_type="estimator",
            estimator_names=["test"],
            ml_task="binary-classification",
            data_source="test",
            pos_label=1,
            strategy="invalid",
        )


def test_different_strategies_and_bins():
    """Test with different strategies and bin counts."""
    # Create data
    X, y = make_classification(n_samples=500, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Create and fit model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Create YPlotData objects
    y_true_data = [
        YPlotData(estimator_name="LogisticRegression", split_index=None, y=y_test)
    ]
    y_pred_proba = model.predict_proba(X_test)
    y_pred_data = [
        YPlotData(estimator_name="LogisticRegression", split_index=None, y=y_pred_proba)
    ]

    # Test with different strategies
    for strategy in ["uniform", "quantile"]:
        for n_bins in [5, 10, 20]:
            # Create the display
            display = CalibrationCurveDisplay._compute_data_for_display(
                y_true=y_true_data,
                y_pred=y_pred_data,
                report_type="estimator",
                estimator_names=["LogisticRegression"],
                ml_task="binary-classification",
                data_source="test",
                pos_label=1,
                n_bins=n_bins,
                strategy=strategy,
            )

            # Check that the display has the right attributes
            assert display.strategy == strategy
            assert display.n_bins == n_bins

            # Check that we have the expected number of bins
            # The actual number of bins may be fewer than requested if some bins
            # are empty. This is expected behavior from scikit-learn's
            # calibration_curve function
            assert len(display.prob_true[1][0]) >= n_bins - 5
            assert len(display.prob_true[1][0]) <= n_bins


def test_multiclass_calibration():
    """Test calibration curves with multiclass classification."""
    # Create multiclass data
    X, y = make_classification(
        n_samples=1000, n_classes=3, n_informative=6, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Create and fit model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Create report
    report = EstimatorReport(
        model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    # Check each class
    for pos_label in range(3):
        # Get the calibration curve display for this class
        display = report.metrics.calibration_curve(pos_label=pos_label)

        # Check that the display is of the right type
        assert isinstance(display, CalibrationCurveDisplay)

        # Verify attributes
        assert display.pos_label == pos_label
        assert display.data_source == "test"
        assert display.ml_task == "multiclass-classification"
        assert display.report_type == "estimator"

        # Test that we can plot without error
        fig, (ax, hist_ax) = plt.subplots(nrows=2, figsize=(8, 8), height_ratios=[2, 1])
        display.plot(ax=ax, hist_ax=hist_ax)
        plt.close(fig)

    # Test plotting multiple classes on the same plot
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 1, height_ratios=[2, 1])
    ax = fig.add_subplot(gs[0])
    ax.set_title("Multiclass Calibration Curves")
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    # Plot each class
    for pos_label in range(3):
        display = report.metrics.calibration_curve(pos_label=pos_label)
        ax.plot(
            display.prob_pred[pos_label][0],
            display.prob_true[pos_label][0],
            marker="o",
            label=f"Class {pos_label}",
        )

    ax.legend()
    plt.close(fig)


def test_comparison_report_integration():
    """Test that the calibration curve works with ComparisonReport."""
    # Create data
    X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Create and fit models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
    }

    # Fit models
    for _name, model in models.items():
        model.fit(X_train, y_train)

    # Create reports for each model
    reports = {}
    for name, model in models.items():
        reports[name] = EstimatorReport(
            model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )

    # Create comparison report
    comparison = ComparisonReport(reports)

    # Get calibration curve from comparison report
    display = comparison.metrics.calibration_curve(pos_label=1)

    assert isinstance(display, CalibrationCurveDisplay)

    assert display.pos_label == 1
    assert display.report_type == "comparison-estimator"
    assert display.ml_task == "binary-classification"
    assert len(display.estimator_names) == len(models)
    assert all(name in display.estimator_names for name in models)

    # Test plotting with GridSpec
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(3, 2, height_ratios=[2, 1, 1])

    # Plot all curves on one axis
    ax_curve = fig.add_subplot(gs[0, :])
    ax_curve.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    # Plot each model's curve
    for i, name in enumerate(models.keys()):
        ax_curve.plot(
            display.prob_pred[1][i], display.prob_true[1][i], marker="o", label=name
        )

    ax_curve.legend()

    # Plot histograms
    for i, name in enumerate(models.keys()):
        row, col = divmod(i, 2)
        ax_hist = fig.add_subplot(gs[row + 1, col])
        ax_hist.hist(display.y_prob[i], range=(0, 1), bins=10)
        ax_hist.set_title(name)
    plt.close(fig)

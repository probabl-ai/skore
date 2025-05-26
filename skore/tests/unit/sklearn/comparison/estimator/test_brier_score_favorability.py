"""Test for Brier score favorability indicator in ComparisonReport.

Tests that the Brier score favorability indicator shows (↘︎) even when some
estimators don't have predict_proba method.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skore import ComparisonReport, EstimatorReport


def test_brier_score_favorability():
    """Test that Brier score favorability indicator is correctly set to (↘︎)
    even when some estimators don't have predict_proba method."""

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Creating estimators: Where LinearSVC doesn't have predict_proba method
    estimators = {
        "LinearSVC": make_pipeline(
            StandardScaler(), LinearSVC(random_state=0, tol=1e-5)
        ),
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(random_state=0)
        ),
        "RandomForestClassifier": RandomForestClassifier(random_state=0),
    }

    estimator_reports = {}
    for name, est in estimators.items():
        report = EstimatorReport(
            est, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
        )
        estimator_reports[name] = report

    comparison_report = ComparisonReport(estimator_reports)

    # Generate metrics report with favorability indicators
    metrics_df = comparison_report.metrics.report_metrics(
        pos_label=1, indicator_favorability=True
    )

    # Check that the Brier score favorability indicator is (↘︎)
    assert "Brier score" in metrics_df.index
    assert "Favorability" in metrics_df.columns
    assert metrics_df.loc["Brier score", "Favorability"] == "(↘︎)"

    # Verify LinearSVC doesn't have Brier score (should be NaN)
    assert pd.isna(metrics_df.loc["Brier score", "LinearSVC"])

    # But other metrics have a valid Brier score
    assert not pd.isna(metrics_df.loc["Brier score", "LogisticRegression"])

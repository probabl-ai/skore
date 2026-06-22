"""
Tests of ComparisonReport which work regardless whether it holds EstimatorReports or
CrossValidationReports.
"""

from io import BytesIO

import joblib
import pytest
import skrub
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from skore import (
    ComparisonReport,
    CrossValidationReport,
    EstimatorReport,
    evaluate,
)
from skore._externals._sklearn_compat import convert_container
from skore._utils._dataframe import _concat_vertical


def test_pickle(tmp_path, report):
    """Check that we can pickle a comparison report."""
    with BytesIO() as stream:
        joblib.dump(report, stream)
        joblib.load(stream)


def test_cross_validation_report_cleaned_up(report):
    """
    When a CrossValidationReport is passed to a ComparisonReport, and computations are
    done on the ComparisonReport, the CrossValidationReport should remain pickle-able.

    Non-regression test for bug found in:
    https://github.com/probabl-ai/skore/pull/1512
    """
    report.metrics.summarize()
    sub_report = next(iter(report.reports_.values()))

    with BytesIO() as stream:
        joblib.dump(sub_report, stream)


@pytest.mark.parametrize("report", [EstimatorReport, CrossValidationReport])
def test_pos_label_mismatch(report):
    """Check that we raise an error when the positive labels are not the same."""
    X, y = make_classification(random_state=0)
    estimators = {"LinearSVC": LinearSVC(), "LogisticRegression": LogisticRegression()}

    if report is EstimatorReport:
        reports = {
            name: EstimatorReport(
                est, X_train=X, X_test=X, y_train=y, y_test=y, pos_label=i
            )
            for i, (name, est) in enumerate(estimators.items())
        }
    else:
        reports = {
            name: CrossValidationReport(est, X=X, y=y, pos_label=i)
            for i, (name, est) in enumerate(estimators.items())
        }

    err_msg = "Expected all estimators to have the same positive label."
    with pytest.raises(ValueError, match=err_msg):
        ComparisonReport(reports)


def test_comparison_report_pos_label_binary():
    """ComparisonReport.pos_label matches shared binary-classification positive
    label."""
    X, y = make_classification(n_classes=2, random_state=0)
    reports = {
        "a": EstimatorReport(
            LinearSVC(random_state=0),
            X_train=X,
            X_test=X,
            y_train=y,
            y_test=y,
            pos_label=0,
        ),
        "b": EstimatorReport(
            LogisticRegression(random_state=0),
            X_train=X,
            X_test=X,
            y_train=y,
            y_test=y,
            pos_label=0,
        ),
    }
    comparison = ComparisonReport(reports)
    assert comparison.pos_label == 0


def test_comparison_report_pos_label_multiclass_is_none():
    """Non-binary tasks store no comparison-level positive label."""
    X, y = make_classification(n_classes=3, n_informative=3, random_state=0)
    reports = {
        "a": CrossValidationReport(
            LogisticRegression(random_state=0), X, y, splitter=2
        ),
        "b": CrossValidationReport(LinearSVC(random_state=0), X, y, splitter=2),
    }
    comparison = ComparisonReport(reports)
    assert comparison.pos_label is None


@pytest.mark.parametrize(
    ("container_types", "concatenate_train_and_test"),
    [
        (("pandas", "series"), False),
        (("pandas", "series"), True),
        (("array", "array"), False),
        (("array", "array"), True),
        (("polars", "polars_series"), False),
        (("polars", "polars_series"), True),
    ],
)
def test_create_estimator_report_from_estimator_reports(
    container_types, concatenate_train_and_test, binary_classification_data
):
    """Test creating an estimator report from a comparison report with
    EstimatorReports."""
    X, y = binary_classification_data
    X = convert_container(X, container_types[0])
    y = convert_container(y, container_types[1])
    X_experiment, X_heldout, y_experiment, y_heldout = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    estimators = {
        "estimator_1": LinearSVC(random_state=42),
        "estimator_2": LogisticRegression(random_state=42),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X_experiment, y_experiment, test_size=0.2, random_state=42, shuffle=False
    )
    comparison_report = ComparisonReport(
        {
            name: EstimatorReport(
                est, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
            )
            for name, est in estimators.items()
        }
    )

    est_report_w_test = comparison_report.create_estimator_report(
        report_key="estimator_2",
        X_test=X_heldout,
        y_test=y_heldout,
        concatenate_train_and_test=concatenate_train_and_test,
    )

    assert isinstance(est_report_w_test, EstimatorReport)
    assert joblib.hash(est_report_w_test.X_test) == joblib.hash(X_heldout)
    assert joblib.hash(est_report_w_test.y_test) == joblib.hash(y_heldout)
    if not concatenate_train_and_test:
        assert joblib.hash(est_report_w_test.X_train) == joblib.hash(X_train)
        assert joblib.hash(est_report_w_test.y_train) == joblib.hash(y_train)
    else:
        expected_X_train = _concat_vertical(X_train, X_test)
        expected_y_train = _concat_vertical(y_train, y_test)
        assert joblib.hash(est_report_w_test.X_train) == joblib.hash(expected_X_train)
        assert joblib.hash(est_report_w_test.y_train) == joblib.hash(expected_y_train)


@pytest.mark.parametrize(
    "container_types",
    [("pandas", "series"), ("array", "array"), ("polars", "polars_series")],
)
def test_create_estimator_report_from_cross_validation_reports(
    container_types, binary_classification_data
):
    """Test creating an estimator report from a comparison report with
    CrossValidationReports."""
    X, y = binary_classification_data
    X = convert_container(X, container_types[0])
    y = convert_container(y, container_types[1])
    X_experiment, X_heldout, y_experiment, y_heldout = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    estimators = {
        "estimator_1": LinearSVC(random_state=42),
        "estimator_2": LogisticRegression(random_state=42),
    }

    reports = {
        name: CrossValidationReport(est, X=X_experiment, y=y_experiment, splitter=2)
        for name, est in estimators.items()
    }

    comparison_report = ComparisonReport(reports)

    est_report_w_test = comparison_report.create_estimator_report(
        report_key="estimator_2", X_test=X_heldout, y_test=y_heldout
    )

    assert isinstance(est_report_w_test, EstimatorReport)
    assert joblib.hash(est_report_w_test.X_train) == joblib.hash(X_experiment)
    assert joblib.hash(est_report_w_test.y_train) == joblib.hash(y_experiment)
    assert joblib.hash(est_report_w_test.X_test) == joblib.hash(X_heldout)
    assert joblib.hash(est_report_w_test.y_test) == joblib.hash(y_heldout)


def test_create_estimator_report_concatenate_true_rejects_cross_validation_report(
    binary_classification_data,
):
    """concatenate_train_and_test is only defined for EstimatorReport entries."""
    X, y = binary_classification_data
    X_experiment, X_heldout, y_experiment, y_heldout = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    reports = {
        "a": CrossValidationReport(
            LogisticRegression(random_state=42), X=X_experiment, y=y_experiment
        ),
        "b": CrossValidationReport(
            LinearSVC(random_state=42), X=X_experiment, y=y_experiment
        ),
    }
    comparison_report = ComparisonReport(reports)
    err_msg = "`concatenate_train_and_test=True` is not supported when"
    with pytest.raises(ValueError, match=err_msg):
        comparison_report.create_estimator_report(
            report_key="a",
            X_test=X_heldout,
            y_test=y_heldout,
            concatenate_train_and_test=True,
        )


def test_create_estimator_report_invalid_name(
    comparison_estimator_reports_binary_classification,
):
    """Test that an error is raised when an invalid estimator name is provided."""
    comparison_report = comparison_estimator_reports_binary_classification

    err_msg = "Estimator with key InvalidEstimator not found in the comparison report."
    with pytest.raises(ValueError, match=err_msg):
        comparison_report.create_estimator_report(
            report_key="InvalidEstimator", X_test=[0], y_test=None
        )


def test_create_estimator_report_skrub_concatenate_train_and_test_raises():
    """Skrub-backed reports cannot use concatenate_train_and_test=True."""
    X, y = make_classification(n_samples=40, random_state=0)
    data_op_a = skrub.X(X).skb.apply(
        LogisticRegression(C=0.5, random_state=0), y=skrub.y(y)
    )
    data_op_b = skrub.X(X).skb.apply(
        LogisticRegression(C=2.0, random_state=0), y=skrub.y(y)
    )
    split = data_op_a.skb.train_test_split(random_state=0)
    learner_a = data_op_a.skb.make_learner()
    learner_b = data_op_b.skb.make_learner()
    comparison = ComparisonReport(
        {
            "model_a": EstimatorReport(
                learner_a, train_data=split["train"], test_data=split["test"]
            ),
            "model_b": EstimatorReport(
                learner_b, train_data=split["train"], test_data=split["test"]
            ),
        }
    )
    err_msg = "Cannot concatenate train and test data when using a skrub"
    with pytest.raises(ValueError, match=err_msg):
        comparison.create_estimator_report(
            report_key="model_a",
            test_data=split["test"],
            concatenate_train_and_test=True,
        )


def test_create_estimator_report_skrub_uses_fitted_estimator_without_refit():
    """Skrub path should pass fit=False and new test_data for held-out evaluation."""
    X, y = make_classification(n_samples=40, random_state=0)
    data_op_a = skrub.X(X).skb.apply(
        LogisticRegression(C=0.5, random_state=0), y=skrub.y(y)
    )
    data_op_b = skrub.X(X).skb.apply(
        LogisticRegression(C=2.0, random_state=0), y=skrub.y(y)
    )
    split = data_op_a.skb.train_test_split(random_state=0)
    learner_a = data_op_a.skb.make_learner()
    learner_b = data_op_b.skb.make_learner()
    source_a = EstimatorReport(
        learner_a, train_data=split["train"], test_data=split["test"]
    )
    comparison = ComparisonReport(
        {
            "model_a": source_a,
            "model_b": EstimatorReport(
                learner_b, train_data=split["train"], test_data=split["test"]
            ),
        }
    )
    final_report = comparison.create_estimator_report(
        report_key="model_a",
        test_data=split["test"],
        concatenate_train_and_test=False,
    )
    assert isinstance(final_report, EstimatorReport)
    assert final_report.fit is False
    assert joblib.hash(final_report.X_train) == joblib.hash(source_a.X_train)
    assert joblib.hash(final_report.X_test) == joblib.hash(source_a.X_test)


def test_create_estimator_report_skrub_requires_test_data():
    """Skrub-backed EstimatorReport entries require test_data."""
    X, y = make_classification(n_samples=40, random_state=0)
    data_op_a = skrub.X(X).skb.apply(
        LogisticRegression(C=0.5, random_state=0), y=skrub.y(y)
    )
    data_op_b = skrub.X(X).skb.apply(
        LogisticRegression(C=2.0, random_state=0), y=skrub.y(y)
    )
    split = data_op_a.skb.train_test_split(random_state=0)
    learner_a = data_op_a.skb.make_learner()
    learner_b = data_op_b.skb.make_learner()
    comparison = ComparisonReport(
        {
            "model_a": EstimatorReport(
                learner_a, train_data=split["train"], test_data=split["test"]
            ),
            "model_b": EstimatorReport(
                learner_b, train_data=split["train"], test_data=split["test"]
            ),
        }
    )
    err_msg = "test_data is required when creating an estimator report from a"
    with pytest.raises(ValueError, match=err_msg):
        comparison.create_estimator_report(report_key="model_a")


def test_create_estimator_report_skrub_rejects_X_test_y_test():
    """Skrub-backed entries must not receive X_test or y_test."""
    X, y = make_classification(n_samples=40, random_state=0)
    data_op_a = skrub.X(X).skb.apply(
        LogisticRegression(C=0.5, random_state=0), y=skrub.y(y)
    )
    data_op_b = skrub.X(X).skb.apply(
        LogisticRegression(C=2.0, random_state=0), y=skrub.y(y)
    )
    split = data_op_a.skb.train_test_split(random_state=0)
    learner_a = data_op_a.skb.make_learner()
    learner_b = data_op_b.skb.make_learner()
    comparison = ComparisonReport(
        {
            "model_a": EstimatorReport(
                learner_a, train_data=split["train"], test_data=split["test"]
            ),
            "model_b": EstimatorReport(
                learner_b, train_data=split["train"], test_data=split["test"]
            ),
        }
    )
    err_msg = "X_test and y_test must be omitted when the source report is"
    with pytest.raises(ValueError, match=err_msg):
        comparison.create_estimator_report(
            report_key="model_a",
            test_data=split["test"],
            X_test=X,
        )


def test_create_estimator_report_tabular_requires_X_test_y_test(
    binary_classification_data,
):
    """Tabular EstimatorReport entries require both X_test and y_test."""
    X, y = binary_classification_data
    X_experiment, X_heldout, y_experiment, y_heldout = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_experiment, y_experiment, test_size=0.2, random_state=42, shuffle=False
    )
    comparison = ComparisonReport(
        {
            "estimator_1": EstimatorReport(
                LinearSVC(random_state=42),
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                LogisticRegression(random_state=42),
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
            ),
        }
    )
    err_msg = "X_test and y_test are required when the source report uses"
    with pytest.raises(ValueError, match=err_msg):
        comparison.create_estimator_report(
            report_key="estimator_2",
            X_test=X_heldout,
            y_test=None,
        )


def test_create_estimator_report_tabular_rejects_test_data(binary_classification_data):
    """Tabular entries must not receive test_data."""
    X, y = binary_classification_data
    X_experiment, X_heldout, y_experiment, y_heldout = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_experiment, y_experiment, test_size=0.2, random_state=42, shuffle=False
    )
    comparison = ComparisonReport(
        {
            "estimator_1": EstimatorReport(
                LinearSVC(random_state=42),
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                LogisticRegression(random_state=42),
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
            ),
        }
    )
    err_msg = "test_data must be omitted when the source report uses tabular"
    with pytest.raises(ValueError, match=err_msg):
        comparison.create_estimator_report(
            report_key="estimator_2",
            X_test=X_heldout,
            y_test=y_heldout,
            test_data={},
        )


@pytest.mark.parametrize(
    "comparison_fixture",
    [
        "comparison_estimator_reports_binary_classification",
        "comparison_cross_validation_reports_binary_classification",
    ],
)
def test_to_markdown(comparison_fixture, request):
    report = request.getfixturevalue(comparison_fixture)
    markdown = report.to_markdown()
    assert markdown.startswith("# ComparisonReport:")
    for section in ("## Estimators", "## Metrics", "## Checks (fast mode)", "## Data"):
        assert section in markdown
    assert "n_rows=" in markdown
    assert "| column | dtype |" in markdown
    for label in report.reports_:
        assert label in markdown
    assert "report.metrics.summarize().frame()" in markdown
    assert "report.checks.summarize()" in markdown
    assert "Mute a check with .checks.summarize(ignore=['<code>'])." in markdown
    assert "| estimator |" not in markdown
    assert "estimator_" not in markdown
    assert "- ml task:" in markdown
    if "cross_validation" in comparison_fixture:
        assert "cross-validation folds" in markdown
        assert "splitter" in markdown
    assert "fit time" in markdown
    assert "predict time" in markdown


def test_to_markdown_pos_label():
    X, y = make_classification(n_classes=2, random_state=0)
    markdown = evaluate(
        [RidgeClassifier(), LogisticRegression()],
        X,
        y,
        pos_label=0,
    ).to_markdown()
    assert "- pos_label: 0" in markdown


def test_to_markdown_different_datasets():
    X1, y = make_classification(n_samples=100, n_features=20, random_state=0)
    X2, _ = make_classification(n_samples=100, n_features=10, random_state=1)
    report = evaluate([LinearRegression(), LinearRegression()], [X1, X2], y)
    markdown = report.to_markdown()
    assert "## Data" in markdown
    assert "| report name | n_rows | n_columns |" in markdown
    assert "| column | dtype |" not in markdown
    for label in report.reports_:
        assert label in markdown


@pytest.mark.parametrize(
    "comparison_fixture",
    [
        "comparison_estimator_reports_binary_classification",
        "comparison_cross_validation_reports_binary_classification",
    ],
)
def test_text_repr(comparison_fixture, request):
    report = request.getfixturevalue(comparison_fixture)
    repr_str = repr(report)
    assert repr_str.startswith("ComparisonReport:")
    assert "to_markdown()" in repr_str
    for label in report.reports_:
        assert label in repr_str
    assert "Accuracy" in repr_str


@pytest.mark.parametrize(
    "comparison_fixture",
    [
        "comparison_estimator_reports_binary_classification",
        "comparison_cross_validation_reports_binary_classification",
        "comparison_estimator_reports_multiclass_classification",
        "comparison_cross_validation_reports_multiclass_classification",
        "comparison_estimator_reports_regression",
        "comparison_cross_validation_reports_regression",
        "comparison_estimator_reports_multioutput_regression",
        "comparison_cross_validation_reports_multioutput_regression",
    ],
)
def test_report_repr_html(comparison_fixture, request):
    report = request.getfixturevalue(comparison_fixture)
    sub_report = next(iter(report.reports_.values()))
    expected_estimator_name = sub_report.estimator_.__class__.__name__
    html_out = report._repr_html_()
    assert "skore-comparison-report-" in html_out
    assert "Model comparison" in html_out
    assert expected_estimator_name in html_out
    assert "skoreInitComparisonReport" in html_out
    assert 'class="tree"' in html_out
    assert "ComparisonReport" in html_out
    assert "docs.skore.probabl.ai" in html_out
    assert "report-tabset" in html_out
    assert "ComparisonReport.metrics" in html_out
    assert "skore-comparison-report-select" in html_out

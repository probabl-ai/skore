from __future__ import annotations

from typing import TYPE_CHECKING, cast

import narwhals as nw
from sklearn.base import clone

from skore._checks.base import Check
from skore._checks.utils import (
    CheckNotApplicable,
    check_score_better_than_baseline,
    collect_scores,
    get_fitted_estimator,
    get_preprocessed_X,
    get_report_y,
    majority_vote,
    split_preprocessor_estimator,
)
from skore._sklearn.feature_names import _get_feature_names
from skore._utils.dataframe import _normalize_y_as_dataframe

if TYPE_CHECKING:
    from skore._checks.utils import MetricKey
    from skore._displays.metrics.metrics_summary_display import MetricsSummaryRow
    from skore._reports.base import _BaseReport
    from skore._reports.cross_validation.report import CrossValidationReport
    from skore._reports.estimator.report import EstimatorReport
    from skore._sklearn.types import EstimatorLike


def _on_par(
    candidate: dict[MetricKey, MetricsSummaryRow],
    reference: dict[MetricKey, MetricsSummaryRow],
) -> bool:
    return majority_vote(
        [
            not check_score_better_than_baseline(
                score=reference[key]["score"],
                baseline=candidate[key]["score"],
                greater_is_better=reference[key]["greater_is_better"],
                floor=0.03,
                fraction=0.10,
            )
            for key in reference.keys() & candidate.keys()
        ]
    )[0]


def _predictor_test_scores(
    report: EstimatorReport | CrossValidationReport,
    predictor: EstimatorLike,
    metric_registry,
    X,
    y,
    X_test=None,
    y_test=None,
) -> dict[MetricKey, MetricsSummaryRow]:
    if report._report_type == "cross-validation":
        from skore._reports.cross_validation.report import CrossValidationReport

        report = CrossValidationReport(
            clone(predictor),
            X=X,
            y=y,
            splitter=report.splitter,
            pos_label=report.pos_label,
            n_jobs=report.n_jobs,
        )
        for sub_report in report.reports_:
            sub_report._metric_registry = metric_registry
        return collect_scores(report, data_source="test")

    from skore._reports.estimator.report import EstimatorReport

    report = EstimatorReport(
        clone(predictor),
        X_train=X,
        y_train=y,
        X_test=X_test,
        y_test=y_test,
        pos_label=report.pos_label,
    )
    report._metric_registry = metric_registry
    return collect_scores(report, data_source="test")


class CheckGoldenFeature(Check):
    """Check for a golden feature (SKD011).

    Detects a single feature that, used alone to refit the estimator, reaches
    scores close to the full model on the report's default predictive metrics.
    Features whose scores also match a model trained on the target itself are
    reported as likely target leakage. Skipped when SKD002 has already flagged
    underfitting.
    """

    code = "SKD011"
    title = "Golden feature"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd011-golden-feature"
    severity = "tip"
    slow = True

    def check_function(self, report: _BaseReport) -> str | None:
        skd002_result = getattr(report, "_check_results_cache", {}).get("SKD002")
        if skd002_result is not None and skd002_result["section"] == "issue":
            raise CheckNotApplicable("Skipped because SKD002 detected underfitting.")

        if report._report_type == "cross-validation":
            report = cast("CrossValidationReport", report)
            X = nw.from_native(get_preprocessed_X(report))
            y = get_report_y(report)
            if X.shape[1] < 2:
                raise CheckNotApplicable("Train data has only one feature.")
            n_features = X.shape[1]
            metric_registry = report.reports_[0]._metric_registry.copy()
            X_train = X_test = X
            y_train = y_test = y
        else:
            report = cast("EstimatorReport", report)
            X_train = nw.from_native(get_preprocessed_X(report, data_source="train"))
            X_test = nw.from_native(get_preprocessed_X(report, data_source="test"))
            y_train = get_report_y(report, data_source="train")
            y_test = get_report_y(report, data_source="test")
            if X_train.shape[1] < 2:
                raise CheckNotApplicable("Train data has only one feature.")
            n_features = X_train.shape[1]
            metric_registry = report._metric_registry
            X, y = X_train, y_train

        preprocessor_, predictor_ = split_preprocessor_estimator(
            get_fitted_estimator(report)
        )
        feature_names = _get_feature_names(
            predictor_,
            transformer=preprocessor_,
            X=X,
            n_features=n_features,
        )
        full_feature_scores = collect_scores(report, data_source="test")

        golden_scores: dict[str, dict[MetricKey, MetricsSummaryRow]] = {}
        for i in range(n_features):
            try:
                if report._report_type == "cross-validation":
                    single_feature_scores = _predictor_test_scores(
                        report,
                        predictor_,
                        metric_registry,
                        X.select(nw.col(feature_names[i])).to_native(),
                        y,
                    )
                else:
                    single_feature_scores = _predictor_test_scores(
                        report,
                        predictor_,
                        metric_registry,
                        X_train.select(nw.col(feature_names[i])).to_native(),
                        y_train,
                        X_test=X_test.select(nw.col(feature_names[i])).to_native(),
                        y_test=y_test,
                    )
            except Exception as exc:
                raise CheckNotApplicable(
                    "Failed to create report from single feature."
                ) from exc
            if _on_par(single_feature_scores, full_feature_scores):
                golden_scores[str(feature_names[i])] = single_feature_scores

        if not golden_scores:
            return None

        def _full_model_message(names: list[str]) -> str:
            return (
                f"A model trained on feature(s) {names} alone has similar "
                "performance to a model trained on all the features, on the default "
                "predictive metrics. This may signal data leakage or excessive "
                "reliance on a single feature."
            )

        try:
            if report._report_type == "cross-validation":
                target_scores = _predictor_test_scores(
                    report,
                    predictor_,
                    metric_registry,
                    _normalize_y_as_dataframe(y),
                    y,
                )
            else:
                target_scores = _predictor_test_scores(
                    report,
                    predictor_,
                    metric_registry,
                    _normalize_y_as_dataframe(y_train),
                    y_train,
                    X_test=_normalize_y_as_dataframe(y_test),
                    y_test=y_test,
                )
        except Exception:
            return _full_model_message(list(golden_scores))

        target_like = [
            name
            for name, scores in golden_scores.items()
            if _on_par(scores, target_scores)
        ]
        proxy = [name for name in golden_scores if name not in target_like]
        messages = []
        if target_like:
            messages.append(
                f"A model trained on feature(s) {target_like} alone has similar "
                "performance to a model trained on the target itself, on the default "
                "predictive metrics. This likely means the target is present among "
                "the features."
            )
        if proxy:
            messages.append(_full_model_message(proxy))
        return "\n".join(messages)

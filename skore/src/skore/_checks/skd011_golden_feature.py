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

if TYPE_CHECKING:
    from skore._reports.base import _BaseReport
    from skore._reports.cross_validation.report import CrossValidationReport
    from skore._reports.estimator.report import EstimatorReport


class CheckGoldenFeature(Check):
    """Check for a golden feature (SKD011).

    Detects a single feature that, used alone to refit the estimator, reaches
    scores close to the full model on the report's default predictive metrics.
    This often signals data leakage or excessive reliance on one feature.

    Note: for skrub learners whose preprocessing vectorizes columns (e.g.
    :class:`~skrub.TableVectorizer`), a raw "golden column" may not appear as a
    single preprocessed feature dimension.
    """

    code = "SKD011"
    title = "Golden feature"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd011-golden-feature"
    severity = "tip"
    slow = True

    def check_function(self, report: _BaseReport) -> str | None:
        if report._report_type == "cross-validation":
            report = cast("CrossValidationReport", report)
            X = nw.from_native(get_preprocessed_X(report))
            y = get_report_y(report)
            if X.shape[1] < 2:
                raise CheckNotApplicable("Train data has only one feature.")
            n_features = X.shape[1]
            metric_registry = report.reports_[0]._metric_registry.copy()
            from skore._reports.cross_validation.report import CrossValidationReport
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
            from skore._reports.estimator.report import EstimatorReport

        preprocessor_, predictor_ = split_preprocessor_estimator(
            get_fitted_estimator(report)
        )
        feature_names = _get_feature_names(
            predictor_,
            transformer=preprocessor_,
            X=X if report._report_type == "cross-validation" else X_train,
            n_features=n_features,
        )
        full_feature_scores = collect_scores(report, data_source="test")

        golden_features: list[str] = []
        single_feature_report: EstimatorReport | CrossValidationReport
        for i in range(n_features):
            try:
                if report._report_type == "cross-validation":
                    single_feature_report = CrossValidationReport(
                        clone(predictor_),
                        X=X.select(nw.col(feature_names[i])).to_native(),
                        y=y,
                        splitter=report.splitter,
                        pos_label=report.pos_label,
                        n_jobs=report.n_jobs,
                    )
                    for sub_report in single_feature_report.reports_:
                        sub_report._metric_registry = metric_registry
                else:
                    single_feature_report = EstimatorReport(
                        clone(predictor_),
                        X_train=X_train.select(nw.col(feature_names[i])).to_native(),
                        y_train=y_train,
                        X_test=X_test.select(nw.col(feature_names[i])).to_native(),
                        y_test=y_test,
                        pos_label=report.pos_label,
                    )
                    single_feature_report._metric_registry = metric_registry
                single_feature_scores = collect_scores(
                    single_feature_report, data_source="test"
                )
            except Exception as exc:
                raise CheckNotApplicable(
                    "Failed to create report from single feature."
                ) from exc
            votes = [
                not check_score_better_than_baseline(
                    score=full_feature_scores[key]["score"],
                    baseline=single_feature_scores[key]["score"],
                    greater_is_better=full_feature_scores[key]["greater_is_better"],
                    floor=0.03,
                    fraction=0.10,
                )
                for key in full_feature_scores.keys() & single_feature_scores.keys()
            ]
            if majority_vote(votes)[0]:
                golden_features.append(str(feature_names[i]))

        if golden_features:
            return (
                f"A model trained on feature(s) {golden_features} alone has similar "
                "performance to a model trained on all the features, on the default "
                "predictive metrics. This may signal data leakage or excessive "
                "reliance on a single feature."
            )
        return None

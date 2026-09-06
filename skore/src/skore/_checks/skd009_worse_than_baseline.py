from __future__ import annotations

from typing import TYPE_CHECKING

from skore._checks.base import Check
from skore._checks.utils import (
    baseline_estimator_report,
    cast_report,
    check_score_better_than_baseline,
    collect_scores,
    majority_vote,
)

if TYPE_CHECKING:
    from skore._reports.base import _BaseReport


class CheckWorseThanBaseline(Check):
    """Check the model's performance against a strong baseline (SKD009).

    Compares test-set scores against a
    :func:`skrub.tabular_pipeline`-wrapped HistGradientBoosting baseline, and
    always reports the baseline's performance: as a warning when the model is
    significantly worse, or for reference when it is on par with or better
    than the baseline.
    """

    code = "SKD009"
    title = "Model performance vs. HistGradientBoosting baseline"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd009-worse-than-baseline"
    severity = "tip"
    slow = True

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast_report(report)
        baseline = baseline_estimator_report(report, kind="performance")

        report_test = collect_scores(report, data_source="test")
        baseline_test = collect_scores(baseline, data_source="test")
        common_keys = sorted(
            report_test.keys() & baseline_test.keys(),
            key=lambda key: tuple(str(part) for part in key),
        )

        # score/baseline are swapped here: we vote when the HGB baseline is
        # significantly better than the model, not when the model fails to beat it.
        worse_votes = [
            check_score_better_than_baseline(
                score=baseline_test[key]["score"],
                baseline=report_test[key]["score"],
                greater_is_better=baseline_test[key]["greater_is_better"],
                floor=0.01,
                fraction=0.05,
            )
            for key in common_keys
        ]
        majority, n_worse, total = majority_vote(worse_votes)

        baseline_scores = []
        for key in common_keys:
            verbose_name, label, average, output = key
            qualifiers = []
            if label is not None:
                qualifiers.append(str(label))
            if average is not None:
                qualifiers.append(average)
            if output is not None:
                qualifiers.append(f"output {output}")
            metric_name = (
                f"{verbose_name} ({', '.join(qualifiers)})"
                if qualifiers
                else verbose_name
            )
            baseline_scores.append(f"{metric_name}={baseline_test[key]['score']:.3g}")
        baseline_performance = ", ".join(baseline_scores)

        if majority:
            return (
                "Test scores are significantly worse than a HistGradientBoosting"
                f" baseline for {n_worse}/{total} default predictive metrics."
                f" Baseline performance on the test set: {baseline_performance}."
            )
        return (
            "Your model is on par with or better than a HistGradientBoosting"
            " baseline. Baseline performance on the test set, for reference:"
            f" {baseline_performance}."
        )

from __future__ import annotations

from typing import TYPE_CHECKING

from skore._checks.base import Check
from skore._checks.utils import CheckNotApplicable, cast_report

if TYPE_CHECKING:
    from skore._reports.base import _BaseReport


class CheckUselessFeatures(Check):
    """Check for useless features (SKD012).

    Flags features whose permutation importance is negligible: either the mean
    importance is negative, below 1e-3, or its interval ``[mean - std,
    mean + std]`` contains zero.

    Permutation importance is computed via the inspection accessor with a
    fixed seed so the result is cached and shared with explicit calls to
    :meth:`~skore.EstimatorReport.inspection.permutation_importance`.
    """

    code = "SKD012"
    title = "Useless features"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd012-useless-features"
    severity = "tip"
    slow = True

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast_report(report)

        try:
            importance_frame = report.inspection.permutation_importance(
                data_source="test", seed=0, n_repeats=5
            ).frame()
        except (ValueError, TypeError) as err:
            raise CheckNotApplicable(
                "Failed to compute permutation importance."
            ) from err

        # group by feature and take the mean over metric/label/output
        per_feature = (
            importance_frame.groupby("feature")[["value_mean", "value_std"]]
            .mean()
            .reset_index()
        )
        mean = per_feature["value_mean"]
        std = per_feature["value_std"]
        useless = per_feature.loc[
            (mean <= 1e-3) | ((mean - std <= 0) & (mean + std >= 0)), "feature"
        ].tolist()
        if useless:
            return (
                f"Feature(s) {useless} have permutation importance overlapping "
                "with zero and could likely be dropped without degrading "
                "performance. Dropping redundant features may also improve model "
                "performance."
            )
        return None

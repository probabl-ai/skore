from __future__ import annotations

from typing import TYPE_CHECKING

from skore._sklearn._diagnostics.base import DiagnosticResult

if TYPE_CHECKING:
    from skore._sklearn._cross_validation.report import CrossValidationReport


def run_cross_validation_diagnostics(
    report: CrossValidationReport,
) -> tuple[list[DiagnosticResult], set[str]]:
    total_splits = len(report.estimator_reports_)
    all_checked_codes: set[str] = set()
    positives_by_code: dict[str, list[DiagnosticResult]] = {}

    for estimator_report in report.estimator_reports_:
        results, checked_codes = estimator_report._get_diagnostics()
        all_checked_codes |= checked_codes
        for result in results:
            positives_by_code.setdefault(result.code, []).append(result)

    aggregated: list[DiagnosticResult] = []
    for code in all_checked_codes:
        positives = positives_by_code.get(code, [])
        if len(positives) > total_splits / 2:
            ref = positives[0]
            aggregated.append(
                DiagnosticResult(
                    code=ref.code,
                    title=ref.title,
                    kind=ref.kind,
                    docs_anchor=ref.docs_anchor,
                    explanation=(
                        f"Detected in {len(positives)}/{total_splits} evaluated splits."
                    ),
                )
            )
    return aggregated, all_checked_codes

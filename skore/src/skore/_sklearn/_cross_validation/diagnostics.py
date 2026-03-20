from __future__ import annotations

from typing import TYPE_CHECKING

from skore._sklearn._diagnostics.base import DiagnosticKind, DiagnosticResult
from skore._sklearn._estimator.diagnostics import (
    OVERFITTING_CODE,
    UNDERFITTING_CODE,
    run_estimator_diagnostics,
)

if TYPE_CHECKING:
    from skore._sklearn._cross_validation.report import CrossValidationReport


def _missing_data_result(
    *,
    code: str,
    title: str,
    kind: DiagnosticKind,
    docs_anchor: str,
) -> DiagnosticResult:
    return DiagnosticResult(
        code=code,
        title=title,
        kind=kind,
        docs_anchor=docs_anchor,
        explanation=(
            "No cross-validation split was available to evaluate this diagnostic."
        ),
        is_issue=False,
        evaluated=False,
    )


def _aggregate_split_diagnostics(
    split_diagnostics: list[DiagnosticResult],
) -> DiagnosticResult:
    diagnostic = split_diagnostics[0]
    evaluated_count = sum(result.evaluated for result in split_diagnostics)
    issue_count = sum(
        result.is_issue for result in split_diagnostics if result.evaluated
    )
    total_count = len(split_diagnostics)
    if evaluated_count == 0:
        return DiagnosticResult(
            code=diagnostic.code,
            title=diagnostic.title,
            kind=diagnostic.kind,
            docs_anchor=diagnostic.docs_anchor,
            explanation="No split could be evaluated for this diagnostic.",
            is_issue=False,
            evaluated=False,
        )
    issue_ratio = issue_count / evaluated_count
    is_issue = issue_ratio > 0.5
    issue_pct = issue_ratio * 100
    missing_count = total_count - evaluated_count
    if is_issue:
        explanation = (
            "Detected in "
            f"{issue_count}/{evaluated_count} evaluated splits ({issue_pct:.0f}%), "
            "which is above the majority threshold."
        )
    elif issue_count == 0:
        explanation = (
            f"No evaluated split triggered this diagnostic (0/{evaluated_count})."
        )
    else:
        explanation = (
            "Detected in "
            f"{issue_count}/{evaluated_count} evaluated splits ({issue_pct:.0f}%), "
            "which is below the majority threshold."
        )
    if missing_count > 0:
        explanation = f"{explanation} {missing_count} split(s) were not evaluated."
    return DiagnosticResult(
        code=diagnostic.code,
        title=diagnostic.title,
        kind=diagnostic.kind,
        docs_anchor=diagnostic.docs_anchor,
        explanation=explanation,
        is_issue=is_issue,
        evaluated=True,
    )


def run_cross_validation_diagnostics(
    report: CrossValidationReport,
    *,
    expensive: bool = False,
) -> list[DiagnosticResult]:
    if not report.estimator_reports_:
        return [
            _missing_data_result(
                code=OVERFITTING_CODE,
                title="Potential overfitting",
                kind="overfitting",
                docs_anchor="skd001-overfitting",
            ),
            _missing_data_result(
                code=UNDERFITTING_CODE,
                title="Potential underfitting",
                kind="underfitting",
                docs_anchor="skd002-underfitting",
            ),
        ]
    split_results: dict[str, list[DiagnosticResult]] = {}
    for estimator_report in report.estimator_reports_:
        for result in run_estimator_diagnostics(
            estimator_report,
            expensive=expensive,
        ):
            split_results.setdefault(result.code, []).append(result)
    return [_aggregate_split_diagnostics(results) for results in split_results.values()]

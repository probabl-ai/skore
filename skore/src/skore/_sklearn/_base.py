from __future__ import annotations

from io import StringIO
from typing import Generic, Literal, TypeVar
from uuid import uuid4

from rich.console import Console
from rich.panel import Panel

from skore._config import configuration
from skore._sklearn._diagnostic.base import Check, DiagnosticDisplay
from skore._sklearn._diagnostic.model_checks import create_model_checks
from skore._sklearn._diagnostic.utils import DiagnosticNotApplicable
from skore._utils.repr.base import AccessorHelpMixin, ReportHelpMixin


class _BaseReport(ReportHelpMixin):
    """Base class for all reports.

    This class centralizes shared report logic (e.g. configuration, accessors) and
    inherits from ``ReportHelpMixin`` to provide a consistent ``help()`` and rich/HTML
    representation across all report types.
    """

    _ACCESSOR_CONFIG: dict[str, dict[str, str]]
    _report_type: Literal[
        "estimator",
        "cross-validation",
        "comparison-estimator",
        "comparison-cross-validation",
    ]

    def _get_issues(self) -> tuple[dict[str, dict], set[str]]:
        """Get the issues from the cache or compute them."""
        for check in self._checks_registry:
            if (
                check.code not in self._issues_cache[1]
                and check.report_type == self._report_type
            ):
                try:
                    self._issues_cache[0].update(check.run(self))
                    self._issues_cache[1].add(check.code)
                except DiagnosticNotApplicable:
                    self._issues_cache[1] |= check.code

        if "cross-validation" in self._report_type or "comparison" in self._report_type:
            aggregated = self._aggregate_checks()
            return (
                self._issues_cache[0] | aggregated[0],
                self._issues_cache[1] | aggregated[1],
            )

        return self._issues_cache

    def diagnose(
        self,
        *,
        ignore: list[str] | tuple[str, ...] | None = None,
    ) -> DiagnosticDisplay:
        """Run checks and return a diagnostic with detected issues.

        Checks look for common modeling problems such as overfitting and
        underfitting. Check codes can be muted per-call via `ignore` or globally
        via :func:`~skore.configuration(ignore_checks=...)` .

        Parameters
        ----------
        ignore : list of str or tuple of str or None, default=None
            Check codes to exclude from the results, e.g. `["SKD001"]`.

        Returns
        -------
        DiagnosticDisplay
            A display object with an HTML representation, with the full list of
            detected issues accessible via the :meth:`~DiagnosticDisplay.frame`
            method.

        Examples
        --------
        >>> from skore import evaluate
        >>> from sklearn.dummy import DummyClassifier
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(random_state=42)
        >>> report = evaluate(DummyClassifier(), X, y, splitter=0.2)
        >>> report.diagnose()
        Diagnostic: 1 issue(s) detected, 2 check(s) ran, 0 ignored.
        - [SKD002] Potential underfitting. Train/test scores are on par and not
        significantly better than the dummy baseline for 8/8 comparable metrics. Read
        our documentation for more details:
        https://docs.skore.probabl.ai/dev/user_guide/diagnostic.html#skd002-underfitting.
        Mute with `ignore=['SKD002']`.
        >>> report.diagnose(ignore=["SKD002"])
        Diagnostic: 0 issue(s) detected, 1 check(s) ran, 1 ignored.
        - No issues were detected in your report!
        """
        ignored: set[str] = set()
        if ignore:
            ignored.update(code.strip().upper() for code in ignore if code.strip())
        if configuration.ignore_checks:
            ignored.update(
                code.strip().upper()
                for code in configuration.ignore_checks
                if code.strip()
            )
        issues, checked_codes = self._get_issues()
        filtered = {
            code: issue for code, issue in issues.items() if code not in ignored
        }
        checks_ran = len(checked_codes - ignored)
        return DiagnosticDisplay(filtered, checks_ran, n_ignored=len(ignored))

    def add_checks(
        self,
        checks: list[Check],
    ) -> None:
        """Add custom diagnostic checks and re-run diagnostics.

        # TODO: write docstring

        Returns
        -------
        DiagnosticDisplay
            The diagnostic display with all issues (built-in + custom).
        """
        self._checks_registry.extend(checks)
        if self._report_type == "cross-validation":
            for report in self.estimator_reports_:
                report.add_checks(checks)
        elif "comparison" in self._report_type:
            for report in self.reports_.values():
                report.add_checks(checks)

    def __init__(self) -> None:
        self.id = uuid4().int
        self._issues_cache: list[dict[str, dict], set[str]] = [{}, set()]
        self._checks_registry: list[Check] = create_model_checks()

    @property
    def _hash(self) -> int:
        # FIXME: only for backward compatibility
        return self.id


ParentT = TypeVar("ParentT", bound="_BaseReport")


class _BaseAccessor(AccessorHelpMixin, Generic[ParentT]):
    """Base class for all accessors.

    Accessors expose additional views on a report (e.g. data, metrics) and inherit from
    ``AccessorHelpMixin`` to provide a dedicated ``help()`` and rich/HTML help tree.
    """

    def __init__(self, parent: ParentT) -> None:
        self._parent = parent

    def _rich_repr(self, class_name: str) -> str:
        """Return a string representation using rich for accessors."""
        string_buffer = StringIO()
        console = Console(file=string_buffer, force_terminal=False)
        console.print(
            Panel(
                "Get guidance using the help() method",
                title=f"[cyan]{class_name}[/cyan]",
                border_style="orange1",
                expand=False,
            )
        )
        return string_buffer.getvalue()

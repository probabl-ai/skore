from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
from io import StringIO
from typing import Generic, Literal, TypeVar
from uuid import uuid4

from rich.console import Console
from rich.panel import Panel

from skore._config import configuration
from skore._sklearn._diagnostic.base import Check, DiagnosticDisplay
from skore._sklearn._diagnostic.model_checks import _BUILTIN_CHECKS
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

    def _aggregate_checks(self) -> tuple[dict[str, dict], set[str]]:
        """Aggregate EstimatorReport checks.

        Overwritten in CrossValidation and Comparison reports.
        """
        return ({}, set())

    def _get_issues(self) -> tuple[dict[str, dict], set[str]]:
        """Get the issues from the cache or compute them."""
        if not hasattr(self, "_issues_cache"):
            self._issues_cache: dict[str, dict] = {}
        if not hasattr(self, "_checked_codes"):
            self._checked_codes: set[str] = set()

        for check in self._checks_registry:
            if (
                check.code not in self._checked_codes
                and check.report_type == self._report_type
            ):
                try:
                    explanation = check.check_function(self)
                    if explanation:
                        self._issues_cache.update(
                            {
                                check.code: {
                                    "title": check.title,
                                    "docs_url": check.docs_url,
                                    "explanation": explanation,
                                }
                            }
                        )
                    self._checked_codes.add(check.code)
                except DiagnosticNotApplicable:
                    self._checked_codes |= {check.code}

        if "cross-validation" in self._report_type or "comparison" in self._report_type:
            aggregated = self._aggregate_checks()
            return (
                self._issues_cache | aggregated[0],
                self._checked_codes | aggregated[1],
            )

        return self._issues_cache, self._checked_codes

    def diagnose(
        self,
        *,
        ignore: list[str] | tuple[str, ...] | None = None,
    ) -> DiagnosticDisplay:
        """Run checks and return a diagnostic with detected issues.

        Checks look for common modeling problems such as overfitting and
        underfitting. Check codes can be muted per-call via `ignore` or globally
        via :func:`~skore.configuration()` with `ignore_checks=...`.

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
        Diagnostic: 1 issue(s) detected, 4 check(s) ran, 0 ignored.
        - [SKD002] Potential underfitting.
        ...
        >>> report.diagnose(ignore=["SKD002"])
        Diagnostic: 0 issue(s) detected, 3 check(s) ran, 1 ignored.
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
        """Register additional diagnostic checks for this report.

        Checks are defined by implementing the :class:`~skore.Check` protocol.

        Appends the given checks to the registry used by
        :meth:`diagnose`. The next call to :meth:`diagnose` runs any newly added
        checks (along with checks that have not yet been cached). Already-run
        built-in checks are not re-executed.

        Parameters
        ----------
        checks : list of Check
            Additional checks to register

        """
        report_types = [
            "cross-validation",
            "estimator",
            "comparison-estimator",
            "comparison-cross-validation",
        ]
        for check in checks:
            if not isinstance(check, Check):
                raise ValueError(f"{check} does not implement the Check protocol.")
            if check.report_type not in report_types:
                raise ValueError(
                    f"Check report_type should be one of: {', '.join(report_types)}. "
                    f"Got {check.report_type} instead."
                )
        self._checks_registry.extend(checks)

    def __init__(self) -> None:
        self._metadata = {
            "id": uuid4().int,
            "skore-version": version("skore"),
            "creation-date": datetime.now(timezone.utc).isoformat(),
            # comparison reports don't have a _report_type yet at init time
            # but they don't have a `get_state` anyway:
            "report_type": getattr(self, "_report_type", "comparison"),
        }
        self._checks_registry: list[Check] = list(_BUILTIN_CHECKS)

    @property
    def id(self):
        return self._metadata["id"]

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

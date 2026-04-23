from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
from io import StringIO
from typing import Generic, Literal, TypeVar
from uuid import uuid4

from rich.console import Console
from rich.panel import Panel

from skore._config import configuration
from skore._sklearn._diagnostic.base import Check, CheckCode, DiagnosticDisplay
from skore._sklearn._diagnostic.model_checks import _BUILTIN_CHECKS
from skore._sklearn._diagnostic.utils import CheckNotApplicable
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

    def _aggregate_checks(self) -> tuple[dict[CheckCode, dict], set[CheckCode]]:
        """Aggregate EstimatorReport checks.

        Overwritten in CrossValidation and Comparison reports.

        Returns ``(findings, applicable_codes)``.
        """
        return ({}, set())

    def _get_findings(self) -> tuple[dict[CheckCode, dict], set[CheckCode]]:
        """Get the findings from the cache or compute them.

        Returns ``(findings, applicable_codes)`` where ``applicable_codes``
        contains the codes of the checks that actually ran on the report,
        i.e. those that did not raise :class:`CheckNotApplicable`.
        """
        if not hasattr(self, "_findings_cache"):
            self._findings_cache: dict[CheckCode, dict] = {}
        if not hasattr(self, "_applicable_codes"):
            self._applicable_codes: set[CheckCode] = set()
        if not hasattr(self, "_not_applicable_codes"):
            self._not_applicable_codes: set[CheckCode] = set()

        for check in self._checks_registry:
            if (
                check.report_type == self._report_type
                and check.code not in self._applicable_codes
                and check.code not in self._not_applicable_codes
            ):
                try:
                    explanation = check.check_function(self)
                except CheckNotApplicable:
                    self._not_applicable_codes.add(check.code)
                    continue
                if explanation:
                    self._findings_cache[check.code] = {
                        "title": check.title,
                        "docs_url": check.docs_url,
                        "explanation": explanation,
                        "severity": getattr(check, "severity", "issue"),
                    }
                self._applicable_codes.add(check.code)

        if "cross-validation" in self._report_type or "comparison" in self._report_type:
            agg_findings, agg_applicable = self._aggregate_checks()
            return (
                self._findings_cache | agg_findings,
                self._applicable_codes | agg_applicable,
            )

        return self._findings_cache, self._applicable_codes

    def diagnose(
        self,
        *,
        ignore: list[CheckCode] | tuple[CheckCode, ...] | None = None,
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
            A display object with an HTML representation organized as three
            tabs (``Issues``, ``Tips``, ``Passed``). The full list of findings
            is accessible via the :meth:`~DiagnosticDisplay.frame` method.

        Examples
        --------
        >>> from skore import evaluate
        >>> from sklearn.dummy import DummyClassifier
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(random_state=42)
        >>> report = evaluate(DummyClassifier(), X, y, splitter=0.2)
        >>> report.diagnose()
        Diagnostic: 1 issue(s), ...
        Issues:
        - [SKD002] Potential underfitting...
        ...
        >>> report.diagnose(ignore=["SKD002"])
        Diagnostic: 0 issue(s), ... 1 ignored.
        ...
        """
        ignored: set[CheckCode] = set()
        if ignore:
            ignored.update(code.strip().upper() for code in ignore if code.strip())
        if configuration.ignore_checks:
            ignored.update(
                code.strip().upper()
                for code in configuration.ignore_checks
                if code.strip()
            )
        findings, applicable_codes = self._get_findings()
        filtered = {
            code: finding for code, finding in findings.items() if code not in ignored
        }
        checks_metadata: dict[CheckCode, dict] = {
            check.code: {
                "title": check.title,
                "docs_url": check.docs_url,
                "severity": getattr(check, "severity", "issue"),
            }
            for check in self._checks_registry
        }
        return DiagnosticDisplay(
            findings=filtered,
            checks_metadata=checks_metadata,
            applicable_codes=applicable_codes,
            ignored_codes=ignored,
        )

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

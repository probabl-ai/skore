from __future__ import annotations

from datetime import UTC, datetime
from functools import partial
from importlib.metadata import version
from typing import TYPE_CHECKING, Generic, Literal, TypeVar
from uuid import uuid4

from skore._project.git import git_commit
from skore._sklearn._checks._utils import CheckNotApplicable
from skore._sklearn._checks.base import Check, CheckCode, CheckResult, CheckSection
from skore._sklearn._checks.model_checks import _BUILTIN_CHECKS
from skore._sklearn.types import DataSource
from skore._utils._progress_bar import track
from skore._utils.repr.base import (
    AccessorHelpMixin,
    ReportHelpMixin,
    render_panel_to_plain_text,
)

if TYPE_CHECKING:
    import pandas as pd

    from skore._sklearn._checks.accessor import _ChecksAccessor


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

    checks: _ChecksAccessor

    def _aggregate_checks(
        self,
        ignored_codes: set[CheckCode],
        *,
        fast_mode: bool = False,
    ) -> dict[CheckCode, CheckResult]:
        """Aggregate EstimatorReport checks.

        Overwritten in Comparison reports.
        """
        return {}

    def _get_checks_results(
        self,
        ignored_codes: set[CheckCode],
        *,
        fast_mode: bool = False,
    ) -> dict[CheckCode, CheckResult]:
        """Run uncached checks and return the checks summary.

        Parameters
        ----------
        ignored_codes : set of CheckCode
            Check codes to exclude from execution, e.g. ``{"SKD001"}``.

        fast_mode : bool, default=False
            When True, skip slow checks that are not already in the cache
            (their `check_function` is never invoked). Cached slow results
            are still surfaced.

        Returns
        -------
        dict of CheckCode to CheckResult
            Summary of every check applicable to the report type with its display
            section.
        """
        if not hasattr(self, "_check_results_cache"):
            self._check_results_cache: dict[CheckCode, CheckResult] = {}

        checks_to_run = [
            check
            for check in self._checks_registry
            if self._report_type in check.report_types
            and check.code not in self._check_results_cache
            and check.code not in ignored_codes
            and not (fast_mode and check.slow)
        ]
        for check in track(
            checks_to_run,
            description="Running checks",
            total=len(checks_to_run),
            disable=fast_mode,
        ):
            try:
                explanation = check.check_function(self)
                section: CheckSection = (
                    getattr(check, "severity", "issue") if explanation else "passed"
                )
            except CheckNotApplicable as exc:
                explanation = exc.args[0] if exc.args else None
                section = "not_applicable"
            self._check_results_cache[check.code] = {
                "title": check.title,
                "docs_url": check.docs_url,
                "explanation": explanation,
                "section": section,
            }

        if "comparison" in self._report_type:
            return self._aggregate_checks(ignored_codes, fast_mode=fast_mode)

        summary: dict[CheckCode, CheckResult] = {}
        for check in self._checks_registry:
            if self._report_type not in check.report_types:
                continue
            code = check.code
            if code in ignored_codes:
                summary[code] = {
                    "title": check.title,
                    "docs_url": check.docs_url,
                    "explanation": None,
                    "section": "ignored",
                }
            elif fast_mode and check.slow and code not in self._check_results_cache:
                summary[code] = {
                    "title": check.title,
                    "docs_url": check.docs_url,
                    "explanation": None,
                    "section": "skipped",
                }
            elif code in self._check_results_cache:
                summary[code] = self._check_results_cache[code]
        return summary

    def _checks_summary_html_fragment(self) -> str:
        """HTML snippet for the checks summary tab in report reprs."""
        return self.checks.summarize(fast_mode=True)._embedded_repr_html()

    def __init__(self) -> None:
        self._metadata = {
            "id": uuid4().int,
            "skore-version": version("skore"),
            "creation-date": datetime.now(UTC).isoformat(),
            # comparison reports don't have a _report_type yet at init time
            # but they don't have a `to_dict` anyway:
            "report_type": getattr(self, "_report_type", "comparison"),
            "git_commit": git_commit(),
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

    def __repr__(self) -> str:
        return render_panel_to_plain_text(self._create_help_panel())

    def _repr_html_(self) -> str:
        return self._create_help_html()

    def _repr_mimebundle_(self, **kwargs):
        return {"text/plain": repr(self), "text/html": self._repr_html_()}


class BaseMetricsAccessor(_BaseAccessor, Generic[ParentT]):
    """Base class for metrics accessor."""

    def __getattr__(self, name):
        """Define custom metric methods dynamically.

        If attribute ``name`` is defined statically, this method will not be called.
        """
        if name in self.available():
            return partial(lambda *args, **kwargs: self.get(name, *args, **kwargs))

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __dir__(self) -> list[str]:
        """Add custom metrics to __dir__ for tab-completion."""
        return list(set(super().__dir__()).union(self.available()))

    def _formatted_summary_frame(
        self,
        *,
        data_source: DataSource = "test",
        metric: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """Metric summary.

        Used for displaying the accessor.
        """
        return self.summarize().frame()

    def __repr__(self) -> str:
        return (
            "Metrics summary:\n"
            f"{self._formatted_summary_frame()!r}\n"
            "Explore available methods with .help()."
        )

    def _repr_html_(self) -> str:
        return (
            "<p>Metrics summary:</p>"
            f"{self._formatted_summary_frame()._repr_html_()}"
            '<p role="note">Explore available methods with '
            "<code>.help()</code>.</p>"
        )

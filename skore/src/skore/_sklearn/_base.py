from __future__ import annotations

from datetime import UTC, datetime
from importlib.metadata import version
from typing import TYPE_CHECKING, Generic, Literal, TypeVar
from uuid import uuid4

from skore._project.git import git_commit
from skore._sklearn._checks._utils import CheckNotApplicable
from skore._sklearn._checks.base import Check, CheckCode
from skore._sklearn._checks.model_checks import _BUILTIN_CHECKS
from skore._utils._progress_bar import track
from skore._utils.repr.base import (
    AccessorHelpMixin,
    ReportHelpMixin,
    render_panel_to_plain_text,
)

if TYPE_CHECKING:
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
    ) -> tuple[dict[CheckCode, dict], set[CheckCode], set[CheckCode]]:
        """Aggregate EstimatorReport checks.

        Overwritten in CrossValidation and Comparison reports.

        Returns ``(check_results, applicable_codes, not_applicable_codes)``.
        """
        return {}, set(), set()

    def _get_results(
        self,
        ignored_codes: set[CheckCode],
        *,
        fast_mode: bool = False,
    ) -> tuple[dict[CheckCode, dict], set[CheckCode], set[CheckCode]]:
        """Get the check results from the cache or compute them.

        Parameters
        ----------
        ignored_codes : set of CheckCode
            Check codes to exclude from the results, e.g. ``{"SKD001"}``.

        fast_mode : bool, default=False
            When True, skip slow checks that are not already in the cache
            (their `check_function` is never invoked). Cached slow results
            are still surfaced.

        Returns ``(check_results, applicable_codes, not_applicable_codes)``
        where ``applicable_codes`` contains the codes of the checks that ran
        without raising :class:`CheckNotApplicable`, and
        ``not_applicable_codes`` contains those that raised it.
        """
        if not hasattr(self, "_check_results_cache"):
            self._check_results_cache: dict[CheckCode, dict] = {}
        if not hasattr(self, "_applicable_codes"):
            self._applicable_codes: set[CheckCode] = set()
        if not hasattr(self, "_not_applicable_codes"):
            self._not_applicable_codes: set[CheckCode] = set()

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
                self._applicable_codes.add(check.code)
            except CheckNotApplicable as exc:
                explanation = exc.args[0] if exc.args else None
                self._not_applicable_codes.add(check.code)
            self._check_results_cache[check.code] = {
                "title": check.title,
                "docs_url": check.docs_url,
                "explanation": explanation,
                "severity": getattr(check, "severity", "issue"),
            }

        if "comparison" in self._report_type:
            agg_check_results, agg_applicable, agg_not_applicable = (
                self._aggregate_checks(ignored_codes, fast_mode=fast_mode)
            )
            return (
                self._check_results_cache | agg_check_results,
                self._applicable_codes | agg_applicable,
                self._not_applicable_codes | agg_not_applicable,
            )

        return (
            self._check_results_cache,
            self._applicable_codes,
            self._not_applicable_codes,
        )

    def _clear_checks_cache(self) -> None:
        if hasattr(self, "_check_results_cache"):
            self._check_results_cache.clear()
        if hasattr(self, "_applicable_codes"):
            self._applicable_codes.clear()
        if hasattr(self, "_not_applicable_codes"):
            self._not_applicable_codes.clear()

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

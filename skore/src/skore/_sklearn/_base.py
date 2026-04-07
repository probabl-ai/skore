from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from io import StringIO
from typing import ClassVar, Generic, Literal, TypeVar
from uuid import uuid4

from rich.console import Console
from rich.panel import Panel

from skore._config import configuration
from skore._sklearn._diagnostic.base import DiagnosticDisplay
from skore._sklearn._diagnostic.utils import (
    DiagnosticNotApplicable,
    validate_check_result,
)
from skore._utils.repr.base import AccessorHelpMixin, ReportHelpMixin


class _BaseReport(ReportHelpMixin):
    """Base class for all reports.

    This class centralizes shared report logic (e.g. configuration, accessors) and
    inherits from ``ReportHelpMixin`` to provide a consistent ``help()`` and rich/HTML
    representation across all report types.
    """

    _ACCESSOR_CONFIG: dict[str, dict[str, str]]
    _BUILTIN_CHECKS: ClassVar[list[tuple[str, Callable]]] = []
    _report_type: Literal[
        "estimator",
        "cross-validation",
        "comparison-estimator",
        "comparison-cross-validation",
    ]
    _issues_cache: tuple[dict[str, dict], set[str]]

    @abstractmethod
    def _run_checks(self) -> tuple[dict[str, dict], set[str]]:
        """Return detected issues and the set of check codes that were evaluated."""

    def _get_issues(self) -> tuple[dict[str, dict], set[str]]:
        """Get the issues from the cache or compute them."""
        if not hasattr(self, "_issues_cache"):
            self._issues_cache = self._run_checks()
        return self._issues_cache

    def _run_own_checks(self) -> tuple[dict[str, dict], set[str]]:
        """Run checks registered on this report, using per-check caching.

        Returns a tuple of (detected issues, set of check codes that were evaluated).
        """
        issues: dict[str, dict] = {}
        checked_codes: set[str] = set()

        for code, check_fn in self._checks:
            if code in self._check_results_cache:
                cached_issues, cached_codes = self._check_results_cache[code]
                issues.update(cached_issues)
                checked_codes |= cached_codes
                continue

            try:
                result = check_fn(self)
                validate_check_result(result)
                issues.update(result)
                result_codes = {code} | set(result.keys())
                checked_codes |= result_codes
                self._check_results_cache[code] = (result, result_codes)
            except DiagnosticNotApplicable:
                checked_codes.add(code)
                self._check_results_cache[code] = ({}, {code})

        return issues, checked_codes

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

    def _resolve_check_targets(self, level: str | None) -> list[_BaseReport]:
        """Return the report instances a check should be registered on.

        Subclasses override this to support propagating checks to sub-reports.
        """
        if level is not None:
            raise ValueError(
                f"level={level!r} is not supported for {type(self).__name__}."
            )
        return [self]

    def add_checks(
        self,
        checks: list[tuple[str, Callable]],
        *,
        level: str | None = None,
    ) -> DiagnosticDisplay:
        """Add custom diagnostic checks and re-run diagnostics.

        Each entry is a ``(code, check_fn)`` pair where *code* is the
        diagnostic code string (e.g. ``"CUSTOM001"``) and *check_fn* is a
        callable accepting a single report argument and returning a dict
        mapping that code to an issue dict with keys ``"title"`` (str),
        ``"explanation"`` (str), and optionally ``"docs_url"`` (str).

        When the check does not detect an issue, it should return an empty dict.

        Raise :class:`~skore._sklearn._diagnostic.utils.DiagnosticNotApplicable`
        inside a check to signal it cannot run on this report.

        Parameters
        ----------
        checks : list of (str, callable)
            Check entries to register. Each tuple pairs a diagnostic code with
            its check callable.

        level : str or None, default=None
            The report level to register checks on. When ``None`` (default),
            checks run on this report directly. For
            :class:`~skore.CrossValidationReport`, pass ``"estimator"`` to
            register on each sub-estimator report (results are aggregated
            across splits). For :class:`~skore.ComparisonReport`, pass
            ``"estimator"`` or ``"cross-validation"`` to propagate to the
            underlying sub-reports.

        Returns
        -------
        DiagnosticDisplay
            The diagnostic display with all issues (built-in + custom).
        """
        targets = self._resolve_check_targets(level)
        for entry in checks:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not callable(entry[1])
            ):
                raise TypeError(
                    f"Each check must be a (code, callable) tuple, got {entry!r}."
                )
            for target in targets:
                target._checks.append(entry)

        for target in targets:
            if hasattr(target, "_issues_cache"):
                del target._issues_cache
        if hasattr(self, "_issues_cache"):
            del self._issues_cache

        return self.diagnose()

    def __init__(self) -> None:
        self.id = uuid4().int
        self._checks: list[tuple[str, Callable]] = list(self._BUILTIN_CHECKS)
        self._check_results_cache: dict[str, tuple[dict[str, dict], set[str]]] = {}

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

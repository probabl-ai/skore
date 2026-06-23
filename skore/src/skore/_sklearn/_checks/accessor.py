from __future__ import annotations

from skore._config import configuration
from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor, _BaseReport
from skore._sklearn._checks.base import Check, CheckCode, ChecksSummaryDisplay


class _ChecksAccessor(_BaseAccessor[_BaseReport], DirNamesMixin):
    """Accessor for checks-related operations.

    You can access this accessor using the `checks` attribute.
    """

    def summarize(
        self,
        *,
        ignore: list[CheckCode] | tuple[CheckCode, ...] | None = None,
        fast_mode: bool = False,
    ) -> ChecksSummaryDisplay:
        """Run checks and return a summary with detected issues.

        Checks look for common modeling problems such as overfitting and
        underfitting. Check codes can be muted per-call via `ignore` or globally
        via :func:`~skore.configuration()` with `ignore_checks=...`.

        Parameters
        ----------
        ignore : list of str or tuple of str or None, default=None
            Check codes to exclude from the results, e.g. `["SKD001"]`.

        fast_mode : bool, default=False
            When True, skip the expensive checks that are not already in the
            cache. Cached slow results from a previous call are still surfaced.

        Returns
        -------
        ChecksSummaryDisplay
            A display object with an HTML representation organized as four
            tabs (``Issues``, ``Tips``, ``Passed``, ``Not Applicable``). The
            full list of results is accessible via the
            :meth:`~ChecksSummaryDisplay.frame` method.

        Examples
        --------
        >>> from skore import evaluate
        >>> from sklearn.dummy import DummyClassifier
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(random_state=42)
        >>> report = evaluate(DummyClassifier(), X, y, splitter=0.2)
        >>> summary = report.checks.summarize()
        >>> "SKD002" in summary.frame()["code"].values
        True
        >>> filtered = report.checks.summarize(ignore=["SKD002"])
        >>> "SKD002" in filtered.frame()["code"].values
        False
        """
        ignored_codes: set[CheckCode] = set()
        if ignore:
            ignored_codes.update(
                code.strip().upper() for code in ignore if code.strip()
            )
        if configuration.ignore_checks:
            ignored_codes.update(
                code.strip().upper()
                for code in configuration.ignore_checks
                if code.strip()
            )
        check_results, applicable_codes, not_applicable_codes = (
            self._parent._get_results(ignored_codes, fast_mode=fast_mode)
        )
        return ChecksSummaryDisplay(
            check_results={
                code: check_result
                for code, check_result in check_results.items()
                if (code in applicable_codes or code in not_applicable_codes)
                and code not in ignored_codes
            },
            not_applicable_codes=not_applicable_codes,
            n_ignored_codes=len(ignored_codes),
            fast_mode=fast_mode,
        )

    def add(
        self,
        checks: list[Check],
    ) -> None:
        """Register additional checks for this report.

        Checks are defined by implementing the :class:`~skore.Check` protocol.

        Parameters
        ----------
        checks : list of Check
            Additional checks to register.
        """
        valid_report_types = {
            "cross-validation",
            "estimator",
            "comparison-estimator",
            "comparison-cross-validation",
        }
        for check in checks:
            if not isinstance(check, Check):
                raise ValueError(f"{check} does not implement the Check protocol.")
            if (
                not isinstance(check.report_type, (list, tuple))
                or not check.report_type
            ):
                raise ValueError(
                    "Check report_type must be a non-empty list of report types."
                )
            invalid_types = set(check.report_type) - valid_report_types
            if invalid_types:
                raise ValueError(
                    f"Check report_type contains unsupported values: "
                    f"{', '.join(sorted(invalid_types))}. "
                    f"Supported values are: {', '.join(sorted(valid_report_types))}."
                )
        self._parent._checks_registry.extend(checks)

    def available(self) -> list[str]:
        """List available checks in the registry.

        Returns
        -------
        list[str]
            The list of available checks in the format "code - title".
        """
        return [
            f"{check.code} - {check.title}" for check in self._parent._checks_registry
        ]

    def remove(self, code: CheckCode) -> None:
        """Remove a check from the registry.

        Parameters
        ----------
        code : str
            The code of the check to remove.
        """
        code = code.strip().upper()
        self._parent._checks_registry = [
            check
            for check in self._parent._checks_registry
            if check.code.upper() != code
        ]
        if hasattr(self._parent, "_check_results_cache"):
            self._parent._check_results_cache.pop(code, None)
        if hasattr(self._parent, "_applicable_codes"):
            self._parent._applicable_codes.discard(code)
        if hasattr(self._parent, "_not_applicable_codes"):
            self._parent._not_applicable_codes.discard(code)

    def __repr__(self) -> str:
        return (
            f"{self.summarize(fast_mode=True)!r}\n"
            "Explore available methods with .help()."
        )

    def _repr_html_(self) -> str:
        return self.summarize(fast_mode=True)._repr_html_()

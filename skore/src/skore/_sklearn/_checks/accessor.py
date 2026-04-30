from __future__ import annotations

from skore._config import configuration
from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor, _BaseReport
from skore._sklearn._checks.base import Check, CheckCode, ChecksSummaryDisplay


class _ChecksAccessor(_BaseAccessor[_BaseReport], DirNamesMixin):
    """Accessor for report checks."""

    def __repr__(self) -> str:
        """Return a string representation using rich."""
        return self._rich_repr(
            class_name=f"skore.{self._parent.__class__.__name__}.checks"
        )

    def summarize(
        self,
        *,
        ignore: list[CheckCode] | tuple[CheckCode, ...] | None = None,
    ) -> ChecksSummaryDisplay:
        """Run checks and return a summary with detected issues.

        Checks look for common modeling problems such as overfitting and
        underfitting. Check codes can be muted per-call via `ignore` or globally
        via :func:`~skore.configuration()` with `ignore_checks=...`.

        Parameters
        ----------
        ignore : list of str or tuple of str or None, default=None
            Check codes to exclude from the results, e.g. `["SKD001"]`.

        Returns
        -------
        ChecksSummaryDisplay
            A display object with an HTML representation organized as three
            tabs (``Issues``, ``Tips``, ``Passed``). The full list of results
            is accessible via the :meth:`~ChecksSummaryDisplay.frame` method.

        Examples
        --------
        >>> from skore import evaluate
        >>> from sklearn.dummy import DummyClassifier
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(random_state=42)
        >>> report = evaluate(DummyClassifier(), X, y, splitter=0.2)
        >>> report.checks.summarize()
        Checks summary: 1 issue(s), ...
        Issues:
        - [SKD002] Potential underfitting...
        ...
        >>> report.checks.summarize(ignore=["SKD002"])
        Checks summary: 0 issue(s), ... 1 ignored.
        ...
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
        check_results, applicable_codes = self._parent._get_results(ignored_codes)
        return ChecksSummaryDisplay(
            check_results={
                code: check_result
                for code, check_result in check_results.items()
                if code in applicable_codes and code not in ignored_codes
            },
            n_ignored_codes=len(ignored_codes),
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
        self._parent._checks_registry.extend(checks)

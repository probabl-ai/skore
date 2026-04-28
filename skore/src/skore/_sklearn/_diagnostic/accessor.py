from __future__ import annotations

from skore._config import configuration
from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor, _BaseReport
from skore._sklearn._diagnostic.base import Check, CheckCode, DiagnosticDisplay


class _DiagnosisAccessor(_BaseAccessor[_BaseReport], DirNamesMixin):
    """Accessor for report diagnostic checks."""

    def __repr__(self) -> str:
        """Return a string representation using rich."""
        return self._rich_repr(
            class_name=f"skore.{self._parent.__class__.__name__}.diagnosis"
        )

    def summarize(
        self,
        *,
        ignore: list[CheckCode] | tuple[CheckCode, ...] | None = None,
    ) -> DiagnosticDisplay:
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
        return DiagnosticDisplay(
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

    def remove(self, *codes: CheckCode) -> None:
        removed_codes = {code.strip().upper() for code in codes if code.strip()}
        if not removed_codes:
            return

        self._parent._checks_registry = [
            check
            for check in self._parent._checks_registry
            if check.code.upper() not in removed_codes
        ]
        if hasattr(self._parent, "_check_results_cache"):
            self._parent._check_results_cache = {
                code: check_result
                for code, check_result in self._parent._check_results_cache.items()
                if code not in removed_codes
            }
        if hasattr(self._parent, "_applicable_codes"):
            self._parent._applicable_codes = {
                code
                for code in self._parent._applicable_codes
                if code not in removed_codes
            }

    def available(self) -> list[Check]:
        return [
            check
            for check in self._parent._checks_registry
            if check.report_type == self._parent._report_type
        ]

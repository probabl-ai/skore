"""Checks that can be run on `skore` reports."""

from typing import TYPE_CHECKING

from skore._externals import lazy_loader

if TYPE_CHECKING:
    from skore._checks.base import Check, ChecksSummaryDisplay
    from skore._checks.utils import CheckNotApplicable

__all__ = ["Check", "ChecksSummaryDisplay", "CheckNotApplicable"]

__getattr__, __dir__, _ = lazy_loader.attach(
    __name__,
    submod_attrs={
        "base": ["Check", "ChecksSummaryDisplay"],
        "builtin": ["_BUILTIN_CHECKS"],
        "utils": ["CheckNotApplicable"],
    },
)

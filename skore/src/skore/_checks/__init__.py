"""Checks that can be run on `skore` reports."""

from typing import TYPE_CHECKING

from skore._externals import lazy_loader

if TYPE_CHECKING:
    from skore._checks.base import Check, ChecksSummaryDisplay
    from skore._checks.utils import CheckNotApplicable

__all__ = ["Check", "ChecksSummaryDisplay", "CheckNotApplicable"]

# Declare objects as importable from here, but lazy-load them to avoid slowdowns.
#
# For an object to be lazy-loaded, declare it:
# - in the ``if TYPE_CHECKING`` block, so type checkers can use it,
# - in ``__all__``, so the F401 linter does not fail,
# - in the ``lazy_loader.attach`` call below.
__getattr__, __dir__, _ = lazy_loader.attach(
    __name__,
    submod_attrs={
        "base": ["Check", "ChecksSummaryDisplay"],
        "builtin": ["_BUILTIN_CHECKS"],
        "utils": ["CheckNotApplicable"],
    },
)

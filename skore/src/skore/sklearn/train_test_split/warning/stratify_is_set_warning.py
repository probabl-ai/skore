"""Stratify warning.

This warning is shown when `stratify` is set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from skore.sklearn.train_test_split.warning.train_test_split_warning import (
    TrainTestSplitWarning,
)

if TYPE_CHECKING:
    ArrayLike = Any


class StratifyWarning(TrainTestSplitWarning):
    """Check whether `stratify` is set."""

    MSG = (
        "We recommend against using `stratify` if you have a problem where your events "
        "occur over time (even if they are independent). Using `stratify` will result "
        "in inflated model performance evaluation because the natural drift of the "
        "target will not be taken into account."
    )

    @staticmethod
    def check(stratify: Optional[ArrayLike], **kwargs) -> Union[str, None]:
        """Check whether stratify is set.

        Parameters
        ----------
        stratify : array-like or None
            An 1-dimensional target array to be used for stratification.

        Returns
        -------
        warning
            None if the check passed, otherwise the warning message.
        """
        if stratify:
            return StratifyWarning.MSG
        return None

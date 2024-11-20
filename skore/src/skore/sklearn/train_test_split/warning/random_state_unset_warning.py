"""'Random state is unset' warning.

This warning is shown when `random_state` is unset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from skore.sklearn.train_test_split.warning.train_test_split_warning import (
    TrainTestSplitWarning,
)

if TYPE_CHECKING:
    from numpy.random import RandomState


class RandomStateUnsetWarning(TrainTestSplitWarning):
    """Check whether `random_state` is set."""

    MSG = (
        "We recommend setting the parameter `random_state`. "
        "This will ensure the reproducibility of your work."
    )

    @staticmethod
    def check(
        shuffle: bool,
        random_state: Optional[Union[int, RandomState]],
        **kwargs,
    ) -> bool:
        """Check whether ``random_state`` is set.

        Parameters
        ----------
        shuffle : bool
            Whether to shuffle the data before splitting.
            If ``False``, the warning is skipped.
        random_state : int or RandomState instance or None
            Controls the shuffling.
            Pass something other than ``None`` to ensure reproducibility.

        Returns
        -------
        bool
            True if the check passed, False otherwise.
        """
        return not (shuffle and random_state is None)

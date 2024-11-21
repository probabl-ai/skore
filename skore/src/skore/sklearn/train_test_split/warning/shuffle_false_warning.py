"""'Shuffle is false' warning.

This warning is shown when `shuffle` is set to False.
"""

from __future__ import annotations

from skore.sklearn.train_test_split.warning.train_test_split_warning import (
    TrainTestSplitWarning,
)


class ShuffleFalseWarning(TrainTestSplitWarning):
    """Check whether ``shuffle`` is set to ``False``."""

    MSG = (
        "We recommend explicitly setting the `shuffle` parameter, in order to show that"
        "this train_test_split is really representative of your production release"
        "process."
    )

    @staticmethod
    def check(
        shuffle: bool,
        **kwargs,
    ) -> bool:
        """Check whether ``shuffle`` is set to ``False``.

        Parameters
        ----------
        shuffle : bool
            Whether to shuffle the data before splitting.

        Returns
        -------
        bool
            True if the check passed, False otherwise.
        """
        return shuffle

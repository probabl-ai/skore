"""'Shuffle is true' warning.

This warning is shown when ``shuffle`` is set to True.
"""

from __future__ import annotations

from skore.sklearn.train_test_split.warning.train_test_split_warning import (
    TrainTestSplitWarning,
)


class ShuffleTrueWarning(TrainTestSplitWarning):
    """Check whether ``shuffle`` is set to ``True``."""

    MSG = (
        "We detected that the `shuffle` parameter is set to `True` either explicitly "
        "or from its default value. In case of time-ordered events (even if they are "
        "independent), this will result in inflated model performance evaluation "
        "because natural drift will not be taken into account. We recommend setting "
        "the shuffle parameter to `False` in order to ensure the evaluation process is "
        "really representative of your production release process."
    )

    @staticmethod
    def check(
        shuffle: bool,
        **kwargs,
    ) -> bool:
        """Check whether ``shuffle`` is set to ``True``.

        Parameters
        ----------
        shuffle : bool
            Whether to shuffle the data before splitting.

        Returns
        -------
        bool
            True if the check passed, False otherwise.
        """
        return shuffle is False

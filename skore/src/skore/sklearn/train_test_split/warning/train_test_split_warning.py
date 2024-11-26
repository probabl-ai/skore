"""Train-test split warning.

This module defines an interface for warnings shown in :func:`~skore.train_test_split`.
"""

from typing import Union


class TrainTestSplitWarning(Warning):
    """Interface for a train-test-split warning."""

    @staticmethod
    def check(*args, **kwargs) -> Union[str, None]:
        """Perform the check.

        Returns
        -------
        warning
            None if the check passed, otherwise the warning message.
        """
        ...

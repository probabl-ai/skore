"""Train-test split warning.

This module defines an interface for warnings shown in :func:`~skore.train_test_split`.
"""


class TrainTestSplitWarning(Warning):
    """Interface for a train-test-split warning."""

    MSG: str

    @staticmethod
    def check(*args, **kwargs) -> bool:
        """Perform the check.

        Returns
        -------
        bool
            True if the check passed, False otherwise.
        """
        ...

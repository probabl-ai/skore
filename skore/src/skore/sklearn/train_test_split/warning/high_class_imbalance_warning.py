"""High class imbalance warning.

This warning is shown when a dataset exhibits a high class imbalance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

import pandas

from skore.sklearn.train_test_split.warning.train_test_split_warning import (
    TrainTestSplitWarning,
)

if TYPE_CHECKING:
    from skore.sklearn.cross_validate import MLTask

    ArrayLike = Any


class HighClassImbalanceWarning(TrainTestSplitWarning):
    """Check whether the test set has high class imbalance."""

    MSG = (
        "It seems that you have a classification problem with a high class "
        "imbalance. In this "
        "case, using train_test_split may not be a good idea because of high "
        "variability in the scores obtained on the test set. "
        "To tackle this challenge we suggest to use skore's "
        "cross_validate function."
    )

    @staticmethod
    def check(
        y: Optional[ArrayLike],
        stratify: Optional[ArrayLike],
        ml_task: MLTask,
        **kwargs,
    ) -> Union[str, None]:
        """Check whether the test set has high class imbalance.

        More precisely, we check whether the most populated class in `y` has
        more than 3 times the size of the least populated class in `y`.
        The other arguments are needed to see if the check is relevant. For
        example, if `y` is a used for a regression task, then the check should
        be skipped.

        Parameters
        ----------
        y : array-like or None
            A 1-dimensional target vector, as a list, numpy array, scipy sparse array,
            or pandas dataframe.
        stratify : array-like or None
            An 1-dimensional target array to be used for stratification.
        ml_task : MLTask
            The type of machine-learning tasks being performed.

        Returns
        -------
        warning
            None if the check passed, otherwise the warning message.
        """
        if stratify or (y is None or len(y) == 0) or ("classification" not in ml_task):
            return None

        class_counts = pandas.Series(y).value_counts()

        if (max(class_counts) / min(class_counts)) < 3:
            return None

        return HighClassImbalanceWarning.MSG

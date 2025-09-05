"""High class imbalance warning.

This warning is shown when a dataset exhibits a high class imbalance.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from skore._sklearn.train_test_split.warning.train_test_split_warning import (
    TrainTestSplitWarning,
)
from skore._sklearn.types import MLTask


class HighClassImbalanceWarning(TrainTestSplitWarning):
    """Check whether the test set has high class imbalance."""

    MSG = (
        "It seems that you have a classification problem with a high class "
        "imbalance. In this "
        "case, using train_test_split may not be a good idea because of high "
        "variability in the scores obtained on the test set. "
        "To tackle this challenge we suggest to use skore's "
        "CrossValidationReport with the `splitter` parameter "
        "of your choice. "
    )

    @staticmethod
    def check(
        y: Sequence | None,
        stratify: ArrayLike | None,
        ml_task: MLTask,
        **kwargs,
    ) -> str | None:
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

        if isinstance(y, np.ndarray) and y.ndim >= 2:
            y = y.flatten()

        counter = Counter(y)
        counts = sorted(counter.values())

        if (counts[-1] / counts[0]) < 3:
            return None

        return HighClassImbalanceWarning.MSG

"""'High class imbalance (too few examples)' warning.

This warning is shown when some class in a dataset has too few examples in the test set.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Optional, Union

from numpy.typing import ArrayLike

from skore.sklearn.train_test_split.warning.train_test_split_warning import (
    TrainTestSplitWarning,
)
from skore.sklearn.types import MLTask


class HighClassImbalanceTooFewExamplesWarning(TrainTestSplitWarning):
    """Check whether the test set has too few examples in one class."""

    MSG = (
        "It seems that you have a classification problem with at least one class with "
        "fewer than 100 examples in the test set. In this case, using train_test_split "
        "may not be a good idea because of high variability in the scores obtained on "
        "the test set. We suggest three options to tackle this challenge: you can "
        "increase test_size, collect more data, or use skore's cross_validate function."
    )

    @staticmethod
    def check(
        y_test: Optional[Sequence],
        stratify: Optional[ArrayLike],
        y_labels: Optional[Sequence],
        ml_task: MLTask,
        **kwargs,
    ) -> Union[str, None]:
        """Check whether the test set has too few examples in one class.

        More precisely, we check whether some class in `y_labels` has
        fewer than 100 examples in `y_test`.
        The other arguments are needed to see if the check is relevant. For
        example, if `y_test` is used for a regression task, then the check should
        be skipped.

        Parameters
        ----------
        y_test : array-like or None
            A 1-dimensional target vector, as a list, numpy array, scipy sparse array,
            or pandas dataframe.
        stratify : array-like or None
            An 1-dimensional target array to be used for stratification.
        y_labels: array-like or None
            A 1-dimensional array containing the class labels
            (e.g. [0, 1] if the task is binary classification),
            if the task is classification.
            This is needed in the case where `y_test` contains zero examples
            for some class.
        ml_task : MLTask
            The type of machine-learning tasks being performed.

        Returns
        -------
        warning
            None if the check passed, otherwise the warning message.
        """
        if (
            stratify
            or (y_test is None or len(y_test) == 0)
            or (y_labels is None or len(y_labels) == 0)
            or ("classification" not in ml_task)
        ):
            return None

        counts = Counter(y_test)

        if all(counts.get(class_, 0) >= 100 for class_ in y_labels):
            return None

        return HighClassImbalanceTooFewExamplesWarning.MSG

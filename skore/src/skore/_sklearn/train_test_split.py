"""Implement train_test_split and TrainTestSplit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.random import RandomState
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    ArrayLike = Any


class TrainTestSplit:
    """Single train-test split implementing the cross-validation protocol.

    This splitter wraps :func:`sklearn.model_selection.train_test_split` and
    exposes ``split`` / ``get_n_splits`` so that it can be passed as the
    ``splitter`` argument of any `skore` or `scikit-learn` function.

    Parameters
    ----------
    test_size : float or int or None, default=0.2
        Proportion (float) or absolute number (int) of samples for the test
        set.  When ``None``, the complement of ``train_size`` is used.

    train_size : float or int or None, default=None
        Proportion (float) or absolute number (int) of samples for the
        training set.  When ``None``, the complement of ``test_size`` is
        used.

    random_state : int, RandomState instance or None, default=0
        Controls the shuffling applied before splitting.  Pass an int for
        reproducible output across multiple calls.

    shuffle : bool, default=True
        Whether to shuffle the data before splitting.

    stratify : array-like or None, default=None
        If not ``None``, data is split in a stratified fashion using this
        as the class labels.

    See Also
    --------
    :func:`sklearn.model_selection.train_test_split` :
        Underlying scikit-learn helper used to generate the split.
    :func:`~skore.train_test_split` :
        Wrapper with additional data-quality warnings.
    :func:`~skore.evaluate` :
        Evaluate an estimator using this splitter via the ``splitter`` parameter.

    Examples
    --------
    >>> import numpy as np
    >>> from skore import TrainTestSplit
    >>> splitter = TrainTestSplit(test_size=0.3, random_state=0)
    >>> X = np.arange(20).reshape(10, 2)
    >>> for train, test in splitter.split(X):
    ...     train, test
    (array([9, 1, 6, 7, 3, 0, 5]), array([2, 8, 4]))
    """

    def __init__(
        self,
        test_size: float | int | None = 0.2,
        train_size: float | int | None = None,
        random_state: int | RandomState | None = 0,
        shuffle: bool = True,
        stratify: ArrayLike | None = None,
    ) -> None:
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify = stratify

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return the number of splits (always 1)."""
        return 1

    def split(self, X: Any, y: Any = None, groups: Any = None):
        """Generate a single train-test split of indices.

        Parameters
        ----------
        X : array-like
            Training data used to determine the number of samples.
        y : array-like or None, default=None
            Ignored, present for API compatibility.
        groups : array-like or None, default=None
            Ignored, present for API compatibility.

        Yields
        ------
        train : ndarray
            The training set indices.
        test : ndarray
            The testing set indices.
        """
        train_idx, test_idx = train_test_split(
            np.arange(X.shape[0] if hasattr(X, "shape") else len(X)),
            test_size=self.test_size,
            train_size=self.train_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=self.stratify,
        )
        yield (train_idx, test_idx)

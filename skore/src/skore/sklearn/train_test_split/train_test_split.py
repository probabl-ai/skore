"""Implement train_test_split."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from numpy.random import RandomState
from rich.panel import Panel

from skore.sklearn.find_ml_task import _find_ml_task
from skore.sklearn.train_test_split.warning import TRAIN_TEST_SPLIT_WARNINGS

if TYPE_CHECKING:
    ArrayLike = Any


def train_test_split(
    *arrays: ArrayLike,
    X: Optional[ArrayLike] = None,
    y: Optional[ArrayLike] = None,
    test_size: Optional[Union[int, float]] = None,
    train_size: Optional[Union[int, float]] = None,
    random_state: Optional[Union[int, RandomState]] = None,
    shuffle: bool = True,
    stratify: Optional[ArrayLike] = None,
):
    """Perform train-test-split of data.

    This is a wrapper over scikit-learn's
    :func:`sklearn.model_selection.train_test_split` helper function,
    enriching it with various warnings.

    The signature is fully compatible with sklearn's ``train_test_split``, and
    some keyword arguments are added to make the detection of issues more accurate.
    For instance, argument ``y`` has been added to pass the target explicitly, which
    makes it easier to detect issues with the target.

    See the :ref:`example_train_test_split` example.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas
        dataframes.
    X : array-like, optional
        If not None, will be appended to the list of arrays passed positionally.
    y : array-like, optional
        If not None, will be appended to the list of arrays passed positionally, after
        ``X``. If None, it is assumed that the last array in ``arrays`` is ``y``.
    test_size : float or int, optional
        If float, should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the test split. If int, represents the absolute number
        of test samples. If None, the value is set to the complement of the train size.
        If train_size is also None, it will be set to 0.25.
    train_size : float or int, optional
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the train split. If int, represents the absolute
        number of train samples. If None, the value is automatically set to the
        complement of the test size.
    random_state : int or numpy RandomState instance, optional
        Controls the shuffling applied to the data before applying the split. Pass an
        int for reproducible output across multiple function calls.
    shuffle : bool, default is True
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.
    stratify : array-like, optional
        If not None, data is split in a stratified fashion, using this as the
        class labels.

    Returns
    -------
    splitting : list
        List containing train-test split of inputs.
        The length of the list is twice the number of arrays passed, including
        the ``X`` and ``y`` keyword arguments. If arrays are passed positionally as well
        as through ``X`` and ``y``, the output arrays are ordered as follows: first the
        arrays passed positionally, in the order they were passed, then ``X`` if it
        was passed, then ``y`` if it was passed.

    Examples
    --------
    >>> # xdoctest: +SKIP
    >>> import numpy as np
    >>> X, y = np.arange(10).reshape((5, 2)), range(5)

    >>> # Drop-in replacement for sklearn train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...     test_size=0.33, random_state=42)
    >>> X_train
    array([[4, 5],
           [0, 1],
           [6, 7]])

    >>> # Explicit X and y, makes detection of problems easier
    >>> X_train, X_test, y_train, y_test = train_test_split(X=X, y=y,
    ...     test_size=0.33, random_state=42)
    >>> X_train
    array([[4, 5],
           [0, 1],
           [6, 7]])

    >>> # When passing X and y explicitly, X is returned before y
    >>> arr = np.arange(10).reshape((5, 2))
    >>> splits = train_test_split(
    ...     arr, y=y, X=X, test_size=0.33, random_state=42)
    >>> arr_train, arr_test, X_train, X_test, y_train, y_test = splits
    >>> X_train
    array([[4, 5],
           [0, 1],
           [6, 7]])
    """
    import sklearn.model_selection

    new_arrays = list(arrays)
    if X is not None:
        new_arrays.append(X)
    if y is not None:
        new_arrays.append(y)

    output = sklearn.model_selection.train_test_split(
        *new_arrays,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
    )

    if X is None:
        X = arrays[0] if len(arrays) == 1 else arrays[-2]

    if y is None and len(arrays) >= 2:
        y = arrays[-1]

    if y is not None:
        y_labels = np.unique(y)
        y_test = output[-1]
    else:
        y_labels = None
        y_test = None

    ml_task = _find_ml_task(y)

    kwargs = dict(
        arrays=new_arrays,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
        X=X,
        y=y,
        y_test=y_test,
        y_labels=y_labels,
        ml_task=ml_task,
    )

    from skore import console  # avoid circular import

    for warning_class in TRAIN_TEST_SPLIT_WARNINGS:
        warning = warning_class.check(**kwargs)

        if warning is not None and (
            not warnings.filters
            or not any(
                f[0] == "ignore" and f[2] == warning_class for f in warnings.filters
            )
        ):
            console.print(
                Panel(
                    title=warning_class.__name__,
                    renderable=warning,
                    style="orange1",
                    border_style="cyan",
                )
            )

    return output

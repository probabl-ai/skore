"""Implement train_test_split."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.random import RandomState
from rich.panel import Panel

from skore._sklearn.find_ml_task import _find_ml_task
from skore._sklearn.train_test_split.warning import TRAIN_TEST_SPLIT_WARNINGS

if TYPE_CHECKING:
    ArrayLike = Any


def train_test_split(
    *arrays: ArrayLike,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    test_size: int | float | None = None,
    train_size: int | float | None = None,
    random_state: int | RandomState | None = None,
    shuffle: bool = True,
    stratify: ArrayLike | None = None,
    as_dict: bool = False,
    **keyword_arrays: ArrayLike,
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
    as_dict : bool, default is False
        If True, returns a Dictionary with keys values ``X_train``, ``X_test``,
        ``y_train``, and ``y_test`` instead of a List. Requires data to be
        passed as keyword arguments, though the first two positional arguments are
        implicitly treated as X and y.
    **keyword_arrays : array-like, optional
        Additional array-like arguments passed by keyword. Used to determine the keys
        of the output dict when ``as_dict=True``.

    Returns
    -------
    splitting : list or dict
        If ``as_dict=False`` (the default): List containing train-test split of inputs.
        The length of the list is twice the number of arrays passed, including
        the ``X`` and ``y`` keyword arguments. If arrays are passed positionally as well
        as through ``X`` and ``y``, the output arrays are ordered as follows: first the
        arrays passed positionally, in the order they were passed, then ``X`` if it
        was passed, then ``y`` if it was passed.

        If ``as_dict=True``: Dictionary with keys
        ``X_train``, ``X_test``, ``y_train``, and ``y_test``,
        each containing respective split data.

    Examples
    --------
    >>> # xdoctest: +SKIP
    >>> import numpy as np
    >>> X, y = np.arange(10).reshape((5, 2)), range(5)

    >>> # Drop-in replacement for sklearn train_test_split
    >>> from skore import train_test_split
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

    >>> # Returns dictionary when as_dict is True, inputs must be keyword arguments.
    >>> sample_weights = np.arange(10).reshape((5, 2))
    >>> split_dict = train_test_split(
    ...     X=X, y=y, sample_weights=sample_weights, as_dict=True, random_state=0)
    >>> split_dict
    {'X_train': ..., 'X_test': ...,
     'y_train': ..., 'y_test': ...,
     'sample_weights_train': ..., 'sample_weights_test': ...}

    >>> # With as_dict is True, the first 2 arguments are implicitly treated as X and y
    >>> train_test_split(X, as_dict=True)
    {'X_train': ..., 'X_test': ...}
    >>> train_test_split(X, y, as_dict=True)
    {'X_train': ..., 'X_test': ...,
     'y_train': ..., 'y_test': ...}
    >>> train_test_split(X, y, sample_weights=sample_weights, as_dict=True)
    {'X_train': ..., 'X_test': ...,
     'y_train': ..., 'y_test': ...,
     'sample_weights_train': ..., 'sample_weights_test': ...}
    """
    import sklearn.model_selection

    new_arrays = list(arrays)
    keys = []

    if X is not None:
        new_arrays.append(X)
        keys.append("X")
    if y is not None:
        new_arrays.append(y)
        keys.append("y")

    if as_dict and arrays:
        # if X or y is passed both by keyword and by position
        if X is not None:
            raise ValueError(
                "With as_dict=True, expected X to be passed either "
                "by position or keyword, not both."
            )
        if y is not None:
            raise ValueError(
                "With as_dict=True, expected y to be passed either "
                "by position or keyword, not both."
            )

        # if X or y are passed by position
        if len(arrays) == 1:
            keys.append("X")
        elif len(arrays) == 2:
            keys.extend(["X", "y"])
        elif len(arrays) > 2:
            raise ValueError(
                "With as_dict=True, expected no more than two positional arguments "
                "(which will be interpreted as X and y). "
                "The remaining arrays must be passed by keyword, "
                "e.g. train_test_split(X, y, z=z, sw=sample_weights, as_dict=True)."
            )

    if keyword_arrays:
        if X is None and y is None:
            arrays = tuple(
                keyword_arrays.values()
            )  # if X and y is not passed but other variables
        new_arrays.extend(keyword_arrays.values())
        keys.extend(keyword_arrays.keys())

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

    y_labels: np.ndarray | None

    if y is not None:
        y_labels = np.unique(y)
        y_test = (
            output[3] if keyword_arrays else output[-1]
        )  # when more kwargs are given
    else:
        y_labels = None
        y_test = None

    ml_task = _find_ml_task(y)

    kwargs = {
        "arrays": new_arrays,
        "test_size": test_size,
        "train_size": train_size,
        "random_state": random_state,
        "shuffle": shuffle,
        "stratify": stratify,
        "X": X,
        "y": y,
        "y_test": y_test,
        "y_labels": y_labels,
        "ml_task": ml_task,
    }

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

    if as_dict:
        result = {}
        for i, k in enumerate(keys):
            train, test = i * 2, i * 2 + 1
            result[f"{k}_train"], result[f"{k}_test"] = output[train], output[test]
        output = result

    return output

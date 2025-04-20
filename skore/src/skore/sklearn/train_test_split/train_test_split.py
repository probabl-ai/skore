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
    as_dict: bool = False,
    **keyword_arrays: ArrayLike,
):
    """Perform train-test-split of data.

    This is a wrapper over scikit-learn's train_test_split helper function,
    enriching it with various warnings and additional functionality.
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

    if as_dict:
        if X is None and y is None:
            if not keyword_arrays:
                raise ValueError(
                    "When as_dict=True, arrays must be passed as keyword arguments"
                )

            new_arrays = list(keyword_arrays.values())

        if X is not None:
            new_arrays.append(X)
            keys.append("X")
        if y is not None:
            new_arrays.append(y)
            keys.append("y")

        keys += list(keyword_arrays.keys())
        new_arrays += list(keyword_arrays.values())

    if not new_arrays:
        raise ValueError("At least one array must be provided")

    # Perform the train-test split using sklearn
    output = sklearn.model_selection.train_test_split(
        *new_arrays,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
    )

    if X is None and len(arrays) >= 1:
        X = arrays[0]

    if y is None and len(arrays) >= 2:
        y = arrays[-1]

    if y is not None:
        y_labels = np.unique(y)
        y_test = output[3] if as_dict else output[-1]
    else:
        y_labels = None
        y_test = None

    # Determine the ML task based on y
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

    # Display any warnings related to train-test split
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

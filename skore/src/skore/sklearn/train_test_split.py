"""Implement train_test_split."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Optional, Union

import pandas
from numpy.random import RandomState

from skore.project import Project

if TYPE_CHECKING:
    from skore.sklearn.cross_validate import MLTask

    ArrayLike = Any


def _find_ml_task(y) -> MLTask:
    """Guess the ML task being addressed based on a target array.

    Parameters
    ----------
    y : numpy.ndarray
        A 1-dimensional target vector.

    Returns
    -------
    Literal["binary-classification", "multiclass-classification",
    "regression", "clustering", "unknown"]
        The guess of the kind of ML task being performed.
    """
    import sklearn.utils.multiclass

    if y is None:
        # NOTE: The task might not be clustering
        return "clustering"

    type_of_target = sklearn.utils.multiclass.type_of_target(y)

    if type_of_target == "binary":
        return "binary-classification"

    if type_of_target == "multiclass":
        return "multiclass-classification"

    if "continuous" in type_of_target:
        return "regression"

    if type_of_target == "unknown":
        return "unknown"

    return "unknown"


class HighClassImbalanceWarning(Warning):
    """Check whether the test set has high class imbalance."""

    MSG = (
        "It seems that you have a classification problem with a high class "
        "imbalance: the test set has less than 100 examples of each class. In this "
        "case, using train_test_split may not be a good idea because of high  "
        "variability in the scores obtained on the test set. We suggest three options "
        "to tackle this challenge: you can increase test_size, collect more data, or "
        "use skore.cross_validate."
    )

    @staticmethod
    def check(
        y: Optional[ArrayLike],
        stratify: Optional[ArrayLike],
        ml_task: MLTask,
        **kwargs,
    ) -> bool:
        """Check whether the test set has high class imbalance.

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
        bool
            True if the check passed, False otherwise.
        """
        if stratify or (y is None or len(y) == 0) or ("classification" not in ml_task):
            return True

        class_counts = pandas.Series(y).value_counts()

        return (max(class_counts) / min(class_counts)) < 3


def train_test_split(
    *arrays: ArrayLike,
    X: Optional[ArrayLike] = None,
    y: Optional[ArrayLike] = None,
    test_size: Optional[Union[int, float]] = None,
    train_size: Optional[Union[int, float]] = None,
    random_state: Optional[Union[int, RandomState]] = None,
    shuffle: bool = True,
    stratify: Optional[ArrayLike] = None,
    project: Optional[Project] = None,
):
    """Perform train-test-split of data.

    This is a layer over scikit-learn's `train_test_split https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html`_
    helper function, enriching it with various warnings that can be saved in a Project.


    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas
        dataframes.
    X : array-like, optional
        If not None, will be appended to the list of arrays passed positionally.
    y : array-like, optional
        If not None, will be appended to the list of arrays passed positionally.
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
    random_state : int or RandomState instance, optional
        Controls the shuffling applied to the data before applying the split. Pass an
        int for reproducible output across multiple function calls. See Glossary.
    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.
    stratify : array-like, optional
        If not None, data is split in a stratified fashion, using this as the
        class labels.
    project : Project, optional
        The project to save information into. If None, no information will be saved.

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.

    Examples
    --------
    >>> import numpy as np
    >>> X, y = np.arange(10).reshape((5, 2)), range(5)

    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...     test_size=0.33, random_state=42)
    >>> X_train
    array([[4, 5],
           [0, 1],
           [6, 7]])

    # Makes detection of problems easier
    >>> X_train, X_test, y_train, y_test = train_test_split(X=X, y=y,
    ...     test_size=0.33, random_state=42)
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

    if len(arrays) >= 2:
        y = arrays[-1]
        y_test = output[-1]
    else:
        y = None
        y_test = None
    ml_task = _find_ml_task(y)

    kwargs = dict(
        arrays=arrays,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
        y=y,
        y_test=y_test,
        ml_task=ml_task,
    )

    for warning_class in [HighClassImbalanceWarning]:
        check = warning_class.check(**kwargs)

        if check is False:
            warnings.warn(
                message=warning_class.MSG, category=warning_class, stacklevel=1
            )

    return output

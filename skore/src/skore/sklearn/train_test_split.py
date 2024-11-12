"""Implement train_test_split."""

from dataclasses import dataclass
from typing import Any, Optional, Union

import pandas
from numpy.random import RandomState

from skore.cross_validate import MLTask
from skore.project import Project

ArrayLike = Any


def _find_ml_task(y) -> MLTask:
    """Guess the ML task being addressed based on a target array.

    Parameters
    ----------
    y : numpy.ndarray, optional
        A target vector.

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


@dataclass
class TrainTestSplitWarningInput:
    """The information obtained during the train-test-split run.

    This information is used for computing various good-practice checks.
    """

    # The train_test_split arguments
    arrays: ArrayLike
    test_size: Optional[Union[int, float]]
    train_size: Optional[Union[int, float]]
    random_state: Optional[Union[int, RandomState]]
    shuffle: bool
    stratify: Optional[ArrayLike]

    # Extra information that was deduced
    y_test: Optional[ArrayLike]
    ml_task: MLTask


class HighClassImbalanceWarning:
    """Check whether the test set has high class imbalance."""

    msg = (
        "It seems that you have a classification problem with a high class "
        "imbalance: the test set has less than 100 examples of each class. In this "
        "case, using train_test_split may not be a good idea because of high  "
        "variability in the scores obtained on the test set. We suggest three options "
        "to tackle this challenge: you can increase test_size, collect more data, or "
        "use skore.cross_validate."
    )

    # @staticmethod
    # def make_inputs(x: TrainTestSplitWarningInput):
    #     y_test = x.y_test
    #     stratify = x.stratify
    #     ml_task = x.ml_task
    #     return y_test, stratify, ml_task

    @staticmethod
    def check(
        y_test=y_test,
        stratify=False,
        ml_task="multiclass-classification",
        # **kwargs
        # x: TrainTestSplitWarningInput
    ) -> bool:
        """Check whether the test set has high class imbalance.

        Returns
        -------
        bool
            True if the check passed, False otherwise
        """
        if (
            (x.stratify is False)
            and (x.y_test is not None)
            and ("classification" in x.ml_task)
        ):
            class_counts = pandas.Series(x.y_test).value_counts()
            if any(count < 100 for count in class_counts) or (
                max(class_counts) / min(class_counts) >= 3
            ):
                return False

        return True


# TODO: warnings in docstring?
def train_test_split(
    *arrays: ArrayLike,
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
    project : Project, optional
        The project to save information into. If None, no information will be saved.
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

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.

    Warnings
    --------

    """
    import sklearn.model_selection

    output = sklearn.model_selection.train_test_split(
        *arrays,
        test_size,
        train_size,
        random_state,
        shuffle,
        stratify,
    )

    if len(arrays) >= 2:
        y = arrays[-1]
        y_test = output[-1]
    else:
        y = None
        y_test = None
    ml_task = _find_ml_task(y)

    x = TrainTestSplitWarningInput(
        arrays=arrays,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
        y_test=y_test,
        ml_task=ml_task,
    )
    failed_checks = []

    for warning_class in [HighClassImbalanceWarning]:
        warning_inputs = warning_class.make_inputs(x)
        check = warning_class.check(*warning_inputs)
        if check is False:
            failed_checks.append(warning_class)

    return output

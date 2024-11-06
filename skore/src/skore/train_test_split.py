"""Implement train_test_split."""

from typing import Any, Literal, Optional, Union

from numpy.random import RandomState

from skore.project import Project

ArrayLike = Any



# TODO: warnings in docstring?
def train_test_split(
    *arrays: ArrayLike,
    project: Optional[Project] = None,
    test_size: Optional[Union[int, float]] = None,
    train_size: Optional[Union[int, float]] = None,
    random_state: Optional[Union[int, RandomState]] = None,
    shuffle: bool = True,
    stratify: Optional[ArrayLike] = None,
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
        test_size=None,
        train_size=None,
        random_state=None,
        shuffle=True,
        stratify=None,
    )

    return output

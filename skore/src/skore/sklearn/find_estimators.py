"""A helper to split a scikit-learn pipeline into estimators."""

import re
from collections import defaultdict

from sklearn.pipeline import FeatureUnion, Pipeline


def find_estimators(pipeline: Pipeline) -> dict[str, list[str]]:
    """
    Split the pipeline into a list of estimators or transformers.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to split.

    Returns
    -------
    dict[str, list[str]]
        A dict of estimators or transformers class in the pipeline,
        grouped by their module name.

    Examples
    --------
    # with a simple scikit-learn pipeline
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.pipeline import make_pipeline
    >>> find_estimators(make_pipeline(LogisticRegression()))

    # with a skrub pipeline
    >>> from skrub import TableVectorizer
    >>> find_estimators(TableVectorizer())
    """
    itemized_pipeline = []
    from sklearn.base import BaseEstimator

    for _, value in pipeline.get_params().items():
        if isinstance(value, BaseEstimator) and not isinstance(value, FeatureUnion):
            itemized_pipeline.append(value)

    classes = set(type(est) for est in itemized_pipeline)

    sklearn_regex = re.compile(r"class \'sklearn\.(\w+)\.")
    skrub_regex = re.compile(r"class \'skrub\.(\w+)\.")

    pipe_steps: dict[str, list[str]] = defaultdict(list)
    for estimator_class in classes:
        class_name = estimator_class.__name__
        estimator_class_str = str(estimator_class)
        sklearn_match = sklearn_regex.search(estimator_class_str)
        skrub_match = skrub_regex.search(estimator_class_str)
        if sklearn_match is not None:
            module = sklearn_match.group(1)
        elif skrub_match is not None:
            module = skrub_match.group(1)
        else:
            module = "other"
        pipe_steps[module].append(class_name)

    # sort, so that even if two pipelines using the same transformers
    # are split in different orders, their dict still match.
    for _, value in pipe_steps.items():
        value.sort()

    return pipe_steps

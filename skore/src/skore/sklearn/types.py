"""Types between parts of the sklearn module."""

from typing import Any, Callable, Literal, Protocol, Union

MLTask = Literal[
    "binary-classification",
    "clustering",
    "multiclass-classification",
    "multioutput-binary-classification",
    "multioutput-multiclass-classification",
    "multioutput-regression",
    "regression",
    "unknown",
]


class SKLearnScorer(Protocol):
    """Protocol defining the interface of scikit-learn's _BaseScorer."""

    _score_func: Callable
    _response_method: Union[str, list[str]]
    _kwargs: dict[str, Any]

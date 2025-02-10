"""Types between parts of the sklearn module."""

from typing import Literal

MLTask = Literal[
    "binary-classification",
    "multiclass-classification",
    "multioutput-regression",
    "regression",
    "clustering",
    "unknown",
]

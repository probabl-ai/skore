"""Types between parts of the sklearn module."""

from typing import Literal

MLTask = Literal[
    "binary-classification",
    "multioutput-binary-classification",
    "multiclass-classification",
    "multioutput-multiclass-classification",
    "multioutput-regression",
    "regression",
    "clustering",
    "unknown",
]

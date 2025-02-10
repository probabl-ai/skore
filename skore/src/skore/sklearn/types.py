"""Types between parts of the sklearn module."""

from typing import Literal

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

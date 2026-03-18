from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeVar

import pandas as pd

DisplayT = TypeVar("DisplayT")


def _iter_child_displays(
    child_displays: Sequence[DisplayT],
    *,
    estimator_names: Sequence[str | None] | None = None,
    split_indices: Sequence[int | None] | None = None,
) -> Iterator[tuple[DisplayT, str | None, int | None]]:
    n_children = len(child_displays)
    if n_children == 0:
        raise ValueError("child_displays must contain at least one display.")

    if estimator_names is None:
        estimator_names = [None] * n_children
    elif len(estimator_names) != n_children:
        raise ValueError("estimator_names must have the same length as child_displays.")

    if split_indices is None:
        split_indices = [None] * n_children
    elif len(split_indices) != n_children:
        raise ValueError("split_indices must have the same length as child_displays.")

    return zip(child_displays, estimator_names, split_indices, strict=True)


def _override_display_metadata(
    frame: pd.DataFrame,
    *,
    estimator_name: str | None,
    split: int | None,
) -> pd.DataFrame:
    frame = frame.copy()
    if estimator_name is not None:
        frame["estimator"] = estimator_name
    if split is not None:
        frame["split"] = split
    return frame

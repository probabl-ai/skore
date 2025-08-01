from typing import Literal

import pandas as pd


def flatten_multi_index(
    index: pd.MultiIndex | pd.Index, join_str: str = "_"
) -> pd.Index:
    """Flatten a pandas MultiIndex into a single-level Index.

    Flatten a pandas `MultiIndex` into a single-level Index by joining the levels
    with the specified string. Empty strings are skipped when joining. Spaces within
    level values are replaced with the join string, and "#" characters are removed
    from the final result.

    Parameters
    ----------
    index : pandas.MultiIndex
        The `MultiIndex` to flatten.

    join_str : str, default="_"
        The string to use for joining the levels.

    Returns
    -------
    pandas.Index
        A flattened `Index` with non-empty levels joined by the specified
        `join_str`. Any spaces within level values are also replaced with
        `join_str`.

    Examples
    --------
    >>> import pandas as pd
    >>> mi = pd.MultiIndex.from_tuples(
    ...     [('a', ''), ('b', '2'), ('c d', 'e f')], names=['letter', 'number']
    ... )
    >>> flatten_multi_index(mi)  # default join_str="_"
    Index(['a', 'b_2', 'c_d_e_f'], dtype='object')
    >>> flatten_multi_index(mi, join_str=" ")
    Index(['a', 'b 2', 'c d e f'], dtype='object')
    >>> flatten_multi_index(mi, join_str="-")
    Index(['a', 'b-2', 'c-d-e-f'], dtype='object')
    """
    if not isinstance(index, pd.MultiIndex):
        return index

    return pd.Index(
        [
            join_str.join(filter(bool, map(str, values)))
            .replace(" ", join_str)
            .replace("#", "")
            for values in index
        ]
    )


def transform_case(
    string: str | None, case_type: Literal["pretty", "snake"]
) -> str | None:
    """Transform the case of a string.

    Parameters
    ----------
    string : str or None
        The string to transform.

    case_type : {"pretty", "snake"}
        The type of case to transform the string to.

    Returns
    -------
    str or None
        The transformed string.
    """
    if string is None:
        return string
    if case_type == "pretty":
        return string.replace("_", " ").capitalize()
    else:  # case_type == "snake"
        return string.replace(" ", "_").lower()


def transform_index(
    index: pd.MultiIndex | pd.Index, case_type: Literal["pretty", "snake"]
) -> pd.MultiIndex | pd.Index:
    """Transform the case of an Index or a MultiIndex.

    Parameters
    ----------
    index : pandas.MultiIndex or pandas.Index
        The index to transform.

    case_type : {"pretty", "snake"}
        The type of case to transform the index to.

    Returns
    -------
    pandas.MultiIndex or pandas.Index
        The transformed index.
    """
    if isinstance(index, pd.MultiIndex):
        new_levels = []
        for level in index.levels:
            new_level = pd.Index(
                [transform_case(str(name), case_type) for name in level]
            )
            new_levels.append(new_level)
        return pd.MultiIndex(
            levels=new_levels,
            codes=index.codes,
            names=[transform_case(name, case_type) for name in index.names],
        )
    else:  # index is a regular Index
        return pd.Index(
            [transform_case(str(name), case_type) for name in index],
            name=transform_case(index.name, case_type),
        )

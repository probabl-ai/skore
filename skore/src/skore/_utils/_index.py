import pandas as pd


def flatten_multi_index(
    index: pd.MultiIndex | pd.Index, join_str: str = "_", *, lowercase: bool = True
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

    lowercase : bool, default=True
        Whether to lowercase the flattened labels.

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
    Index(['a', 'b_2', 'c_d_e_f'], dtype=...)
    >>> flatten_multi_index(mi, join_str=" ")
    Index(['a', 'b 2', 'c d e f'], dtype=...)
    >>> flatten_multi_index(mi, join_str="-")
    Index(['a', 'b-2', 'c-d-e-f'], dtype=...)
    """
    if not isinstance(index, pd.MultiIndex):
        return index

    flattened = [
        (
            join_str.join(filter(bool, map(str, values)))
            .replace(" ", join_str)
            .replace("#", "")
        )
        for values in index
    ]
    if lowercase:
        flattened = [label.lower() for label in flattened]
    return pd.Index(flattened)


def squeeze_single_column(df: pd.DataFrame) -> pd.DataFrame | pd.Series:
    """Convert a single-column DataFrame into a named series."""
    if df.shape[1] != 1:
        return df
    name = (
        flatten_multi_index(df.columns, lowercase=False)[0]
        if isinstance(df.columns, pd.MultiIndex)
        else df.columns[0]
    )
    return df.iloc[:, 0].rename(name)

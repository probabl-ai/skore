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
    Index(['a', 'b_2', 'c_d_e_f'], dtype=...)
    >>> flatten_multi_index(mi, join_str=" ")
    Index(['a', 'b 2', 'c d e f'], dtype=...)
    >>> flatten_multi_index(mi, join_str="-")
    Index(['a', 'b-2', 'c-d-e-f'], dtype=...)
    """
    if not isinstance(index, pd.MultiIndex):
        return index

    return pd.Index(
        [
            join_str.join(filter(bool, map(str, values)))
            .replace(" ", join_str)
            .replace("#", "")
            .lower()
            for values in index
        ]
    )


def rename_columns_and_index(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename names of columns and index names.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame(
            [1, 2, 3],
            index=pd.MultiIndex.from_tuples(
                [(0, ''), (1, ''), (2, '')],
                names=['a', 'b'],
            ),
            columns=["c"],
        )
    >>> frame
         c
    a b
    0    1
    1    2
    2    3
    >>> rename_columns_and_index(frame, {"a": "d", "b": "e", "c": "f"})
         f
    d e
    0    1
    1    2
    2    3
    """
    df_ = df.rename(columns=mapping)
    df_.index = df_.index.set_names(
        [mapping.get(name, name) for name in df_.index.names]
    )
    return df_

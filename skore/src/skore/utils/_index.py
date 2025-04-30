import pandas as pd


def flatten_multi_index(index: pd.MultiIndex) -> pd.Index:
    """Flatten a pandas MultiIndex into a single-level Index.

    Flatten a pandas `MultiIndex` into a single-level Index by joining the levels
    with underscores. Empty strings are skipped when joining. Spaces are replaced by
    an underscore and "#" are skipped.

    Parameters
    ----------
    index : pandas.MultiIndex
        The `MultiIndex` to flatten.

    Returns
    -------
    pandas.Index
        A flattened `Index` with non-empty levels joined by underscores.

    Examples
    --------
    >>> import pandas as pd
    >>> mi = pd.MultiIndex.from_tuples(
    ...     [('a', ''), ('b', '2')], names=['letter', 'number']
    ... )
    >>> flatten_multi_index(mi)
    Index(['a', 'b_2'], dtype='object')
    """
    if not isinstance(index, pd.MultiIndex):
        raise ValueError("`index` must be a MultiIndex.")

    return pd.Index(
        [
            "_".join(filter(bool, map(str, values)))
            .replace(" ", "_")
            .replace("#", "")
            .lower()
            for values in index
        ]
    )

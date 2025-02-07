import pandas as pd


def flatten_multiindex(index: pd.MultiIndex) -> pd.Index:
    """Flatten a pandas MultiIndex into a single-level Index.

    Flatten a pandas MultiIndex into a single-level Index by joining the levels
    with underscores. Empty strings are skipped when joining.

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
    >>> flatten_multiindex(mi)
    Index(['a', 'b_2'], dtype='object')
    """
    if not isinstance(index, pd.MultiIndex):
        raise ValueError("`index` must be a MultiIndex.")

    return pd.Index(["_".join(filter(bool, map(str, values))) for values in index])


def unflatten_index(index: pd.Index, names: list[str] | None = None) -> pd.MultiIndex:
    """Create a MultiIndex from a flat Index with underscore-separated values.

    Convert a flat `Index` with underscore-separated values into a `MultiIndex`.

    Parameters
    ----------
    index : pandas.Index
        The flat Index with values separated by underscores.
    names : list of str, optional
        Names for the levels in the resulting MultiIndex. If None, levels will
        be unnamed.

    Returns
    -------
    pandas.MultiIndex
        A MultiIndex with separate levels for each underscore-separated component.

    Examples
    --------
    >>> import pandas as pd
    >>> flat_idx = pd.Index(['a_1', 'b_2'])
    >>> unflatten_index(flat_idx, names=['letter', 'number'])
    MultiIndex([('a', '1'),
               ('b', '2')],
              names=['letter', 'number'])
    """
    if isinstance(index, pd.MultiIndex):
        raise ValueError("`index` must be a flat Index.")

    tuples = [tuple(val.split("_")) for val in index]
    return pd.MultiIndex.from_tuples(tuples, names=names)

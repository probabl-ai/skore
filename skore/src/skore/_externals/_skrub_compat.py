import pandas as pd
import skrub
from sklearn.utils.fixes import parse_version
from skrub import _dataframe as sbd
from skrub import _join_utils
from skrub._dataframe._common import dispatch

pandas_version = parse_version(parse_version(pd.__version__).base_version)
skrub_version = parse_version(parse_version(skrub.__version__).base_version)


if skrub_version < parse_version("0.6.0"):
    tabular_pipeline = skrub.tabular_learner

    @dispatch
    def concat(*dataframes, axis=0):
        raise NotImplementedError()

    @concat.specialize("pandas", argument_type="DataFrame")
    def _concat_pandas(*dataframes, axis=0):
        kwargs = {"copy": False} if pandas_version < parse_version("3.0") else {}
        if axis == 0:
            return pd.concat(dataframes, axis=0, ignore_index=True, **kwargs)
        else:  # axis == 1
            init_index = dataframes[0].index
            dataframes = [df.reset_index(drop=True) for df in dataframes]
            dataframes = _join_utils.make_column_names_unique(*dataframes)
            result = pd.concat(dataframes, axis=1, **kwargs)
            result.index = init_index
            return result

    @concat.specialize("polars", argument_type="DataFrame")
    def _concat_polars(*dataframes, axis=0):
        import polars as pl

        if axis == 0:
            return pl.concat(dataframes, how="diagonal_relaxed")
        else:  # axis == 1
            dataframes = _join_utils.make_column_names_unique(*dataframes)
            return pl.concat(dataframes, how="horizontal")

    sbd.concat = concat

else:
    tabular_pipeline = skrub.tabular_pipeline


@dispatch
def to_frame(col):
    """Convert a single Column to a DataFrame."""
    raise NotImplementedError()


@to_frame.specialize("pandas", argument_type="Column")
def _to_frame_pandas(col):
    return col.to_frame()


@to_frame.specialize("polars", argument_type="Column")
def _to_frame_polars(col):
    return col.to_frame()


@dispatch
def is_in(col, values):
    raise NotImplementedError()


@is_in.specialize("pandas", argument_type="Column")
def _is_in_pandas(col, values):
    return col.isin(values)


@is_in.specialize("polars", argument_type="Column")
def _is_in_polars(col, values):
    return col.is_in(values)


sbd.to_frame = to_frame
sbd.is_in = is_in

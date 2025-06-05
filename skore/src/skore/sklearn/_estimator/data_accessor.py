import pandas as pd
from sklearn.utils.fixes import parse_version
from skrub import _join_utils
from skrub._dataframe._common import dispatch

from skore.externals._pandas_accessors import DirNamesMixin
from skore.sklearn._base import _BaseAccessor
from skore.sklearn._estimator.report import EstimatorReport
from skore.sklearn._plot import TableReportDisplay

pandas_version = parse_version(parse_version(pd.__version__).base_version)


# TODO: remove when skrub 0.7.0 is released
@dispatch
def concat(*dataframes, axis=0):
    raise NotImplementedError()


@concat.specialize("pandas")
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


@concat.specialize("polars")
def _concat_polars(*dataframes, axis=0):
    import polars as pl

    if axis == 0:
        return pl.concat(dataframes, how="diagonal_relaxed")
    else:  # axis == 1
        dataframes = _join_utils.make_column_names_unique(*dataframes)
        return pl.concat(dataframes, how="horizontal")


class _DataAccessor(_BaseAccessor[EstimatorReport], DirNamesMixin):
    def __init__(self, parent: EstimatorReport) -> None:
        super().__init__(parent)

    def analyze(self, dataset: str, with_y: bool = True) -> TableReportDisplay:
        """Analyse.

        Returns
        -------
        analyzed : str
        """
        if dataset not in (options := ("train", "test", "all")):
            raise ValueError(f"'dataset' options are {options!r}, got {dataset}.")

        X = self._parent.X_train
        X_test = getattr(self._parent, "X_test", None)
        y = getattr(self._parent, "y_train", None)
        y_test = getattr(self._parent, "y_test", None)

        if dataset == "all":
            X = self._parent.X_train
            if X_test is not None:
                X = concat(X, X_test, axis=0)
            if y is not None and y_test is not None:
                y = concat(y, y_test, axis=0)

        if with_y and y is not None:
            X = concat(X, y, axis=1)

        return TableReportDisplay.from_frame(X)

    ####################################################################################
    # Methods related to the help tree
    ####################################################################################

    def _format_method_name(self, name: str) -> str:
        return f"{name}(...)".ljust(29)

    def _get_help_panel_title(self) -> str:
        return "[bold cyan]Available data methods[/bold cyan]"

    def _get_help_tree_title(self) -> str:
        return "[bold cyan]report.data[/bold cyan]"

    def __repr__(self) -> str:
        """Return a string representation using rich."""
        return self._rich_repr(class_name="skore.EstimatorReport.data")

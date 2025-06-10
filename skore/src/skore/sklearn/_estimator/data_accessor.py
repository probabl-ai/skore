from skore.externals._pandas_accessors import DirNamesMixin
from skore.sklearn._base import _BaseAccessor
from skore.sklearn._estimator.report import EstimatorReport
from skore.sklearn._plot import TableReportDisplay
from skore.skrub._skrub_compat import _to_frame_if_column, concat


class _DataAccessor(_BaseAccessor[EstimatorReport], DirNamesMixin):
    def __init__(self, parent: EstimatorReport) -> None:
        super().__init__(parent)

    def analyze(self, dataset: str = "all", with_y: bool = True) -> TableReportDisplay:
        """Plot dataset statistics.

        TODO
        """
        if dataset not in (options := ("train", "test", "all")):
            raise ValueError(f"'dataset' options are {options!r}, got {dataset}.")

        X = self._parent.X_train
        X_test = getattr(self._parent, "X_test", None)
        y = getattr(self._parent, "y_train", None)
        y_test = getattr(self._parent, "y_test", None)

        if dataset == "test":
            X, y = X_test, y_test
        elif dataset == "all":
            if X_test is not None:
                X = concat(X, X_test, axis=0)
            if y is not None and y_test is not None:
                y = concat(
                    _to_frame_if_column(y),
                    _to_frame_if_column(y_test),
                    axis=0,
                )

        if with_y and y is not None:
            X = concat(X, _to_frame_if_column(y), axis=1)

        return TableReportDisplay._compute_data_for_display(X)

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

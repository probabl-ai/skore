from skrub import _dataframe as sbd

from skore.externals._pandas_accessors import DirNamesMixin
from skore.sklearn._base import _BaseAccessor
from skore.sklearn._estimator.report import EstimatorReport
from skore.sklearn._plot import TableReportDisplay
from skore.skrub._skrub_compat import _to_frame_if_column


def _subsample(df, subsample, subsample_strategy, seed):
    if subsample is None:
        return df
    if subsample_strategy == "head":
        return sbd.head(df, subsample)
    elif subsample_strategy == "random":
        return sbd.sample(df, subsample, seed=seed)
    else:
        raise ValueError(
            "'subsample_strategy' options are 'head', 'random', "
            f"got {subsample_strategy}."
        )


class _DataAccessor(_BaseAccessor[EstimatorReport], DirNamesMixin):
    def __init__(self, parent: EstimatorReport) -> None:
        super().__init__(parent)

    def analyze(
        self,
        source_dataset: str = "all",
        with_y: bool = True,
        subsample: int | None = None,
        subsample_strategy: str = "head",
        seed: int | None = None,
    ) -> TableReportDisplay:
        """Plot dataset statistics.

        Parameters
        ----------
        source_dataset : {'train', 'test', 'all'}, default='all'
            The dataset to analyze. If 'train', only the training set is used.
            If 'test', only the test set is used. If 'all', both sets are concatenated
            vertically.

        with_y : bool, default=True
            Whether to include the target variable in the analysis. If True, the target
            variable is concatenated horizontally to the features.

        subsample : int, default=None
            The number of points to subsample the dataframe hold by the display, using
            the strategy set by ``subsample_strategy``. It must be a strictly positive
            integer. If ``None``, no subsampling is applied.

        subsample_strategy : {'head', 'random'}, default='head',
            The strategy used to subsample the dataframe hold by the display. It only
            has an effect when ``subsample`` is not None.

            - If ``'head'``: subsample by taking the ``subsample`` first points of the
              dataframe, similar to Pandas: ``df.head(n)``.
            - If ``'random'``: randomly subsample the dataframe by using a uniform
              distribution. The random seed is controlled by ``random_state``.

        seed : int, default=None
            The random seed to use when randomly subsampling. It only has an effect when
            ``subsample`` is not None and ``subsample_strategy='random'``.

        Returns
        -------
        TableReportDisplay
            A display object containing the dataset statistics and plots.
        """
        if source_dataset not in (options := ("train", "test", "all")):
            raise ValueError(
                f"'dataset' options are {options!r}, got {source_dataset}."
            )

        X = self._parent.X_train
        X_test = getattr(self._parent, "X_test", None)
        y = getattr(self._parent, "y_train", None)
        y_test = getattr(self._parent, "y_test", None)

        if source_dataset == "test":
            X, y = X_test, y_test
        elif source_dataset == "all":
            if X_test is not None:
                X = sbd.concat(X, X_test, axis=0)
            if y is not None and y_test is not None:
                y = sbd.concat(
                    _to_frame_if_column(y),
                    _to_frame_if_column(y_test),
                    axis=0,
                )

        if with_y and y is not None:
            X = sbd.concat(X, _to_frame_if_column(y), axis=1)

        X = _subsample(X, subsample, subsample_strategy, seed)

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

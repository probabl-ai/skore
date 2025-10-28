from typing import Literal

import pandas as pd
from skrub import _dataframe as sbd

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor
from skore._sklearn._cross_validation.report import CrossValidationReport
from skore._sklearn._plot import TableReportDisplay


class _DataAccessor(_BaseAccessor[CrossValidationReport], DirNamesMixin):
    """
    The data accessor helps you to get insights about the dataset used.

    It provides methods to create plots and to visualise the dataset.
    """

    def __init__(self, parent: CrossValidationReport) -> None:
        super().__init__(parent)

    def _retrieve_data_as_frame(
        self,
        with_y: bool,
    ):
        """Retrieve data as DataFrame.

        Parameters
        ----------
        with_y : bool
            Whether we should check that `y` is not None.

        Returns
        -------
        X : DataFrame
            The input data.

        y : DataFrame or None
            The target data.
        """
        X = self._parent.X
        y = self._parent.y

        if not sbd.is_dataframe(X):
            X = pd.DataFrame(X, columns=[f"Feature {i}" for i in range(X.shape[1])])  # type: ignore

        if with_y:
            if y is None:
                raise ValueError("y is required when `with_y=True`.")

            if isinstance(y, pd.Series):
                name = y.name if y.name is not None else "Target"
                y = y.to_frame(name=name)
            elif not sbd.is_dataframe(y):
                if y.ndim == 1:  # type: ignore
                    columns = ["Target"]
                else:
                    columns = [f"Target {i}" for i in range(y.shape[1])]  # type: ignore
                y = pd.DataFrame(y, columns=columns)

        return X, y

    def analyze(
        self,
        with_y: bool = True,
        subsample: int | None = None,
        subsample_strategy: Literal["head", "random"] = "head",
        seed: int | None = None,
    ) -> TableReportDisplay:
        """Plot dataset statistics.

        Parameters
        ----------
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

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = CrossValidationReport(classifier, X=X, y=y, pos_label=1)
        >>> report.data.analyze().frame()
        """
        if subsample_strategy not in (subsample_strategy_options := ("head", "random")):
            raise ValueError(
                f"'subsample_strategy' options are {subsample_strategy_options!r}, got "
                f"{subsample_strategy}."
            )

        X, y = self._retrieve_data_as_frame(with_y)

        df = sbd.concat(X, y, axis=1) if with_y else X

        if subsample:
            if subsample_strategy == "head":
                df = sbd.head(df, subsample)
            else:  # subsample_strategy == "random":
                df = sbd.sample(df, subsample, seed=seed)

        return TableReportDisplay._compute_data_for_display(df)

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
        return self._rich_repr(class_name="skore.CrossValidationReport.data")

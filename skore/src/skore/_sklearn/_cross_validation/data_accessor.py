from typing import Literal

from skrub import _dataframe as sbd

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor
from skore._sklearn._cross_validation.report import CrossValidationReport
from skore._sklearn._plot import TableReportDisplay
from skore._utils._dataframe import _normalize_X_as_dataframe, _normalize_y_as_dataframe


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

        X = _normalize_X_as_dataframe(X)

        if with_y:
            if y is None:
                raise ValueError("y is required when `with_y=True`.")

            y = _normalize_y_as_dataframe(y)

        return X, y

    def _prepare_dataframe_for_display(
        self,
        *,
        with_y: bool | None = None,
        subsample: int | None = None,
        subsample_strategy: Literal["head", "random"] = "head",
        seed: int | None = None,
    ):
        """Return features (and target when ``with_y``) as a single DataFrame.

        When ``with_y`` is ``None``, it defaults to supervised tasks only
        (``ml_task != "clustering"``): for clustering, only ``X`` is returned.
        """
        if subsample_strategy not in (subsample_strategy_options := ("head", "random")):
            raise ValueError(
                f"'subsample_strategy' options are {subsample_strategy_options!r}, got "
                f"{subsample_strategy}."
            )

        if with_y is None:
            with_y = self._parent.ml_task != "clustering"

        X, y = self._retrieve_data_as_frame(with_y)
        df = sbd.concat(X, y, axis=1) if with_y else X

        if subsample:
            if subsample_strategy == "head":
                df = sbd.head(df, subsample)
            else:  # subsample_strategy == "random":
                df = sbd.sample(df, subsample, seed=seed)

        return df

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
        :class:`TableReportDisplay`
            A display object containing the dataset statistics and plots.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression()
        >>> report = evaluate(classifier, X, y, splitter=2)
        >>> report.data.analyze().frame()
        """
        df = self._prepare_dataframe_for_display(
            with_y=with_y,
            subsample=subsample,
            subsample_strategy=subsample_strategy,
            seed=seed,
        )
        return TableReportDisplay._compute_data_for_display(df)

    ####################################################################################
    # Methods related to the help tree
    ####################################################################################

    def __repr__(self) -> str:
        """Return a string representation using rich."""
        return self._rich_repr(class_name="skore.CrossValidationReport.data")

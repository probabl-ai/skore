from typing import Literal

import pandas as pd
from skrub import _dataframe as sbd

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor
from skore._sklearn._estimator.report import EstimatorReport
from skore._sklearn._plot import TableReportDisplay


class _DataAccessor(_BaseAccessor[EstimatorReport], DirNamesMixin):
    """
    The data accessor helps you to get insights about the train and test datasets.

    It provides methods to create plots and to visualise the datasets.
    """

    def __init__(self, parent: EstimatorReport) -> None:
        super().__init__(parent)

    def _retrieve_data_as_frame(
        self,
        dataset: Literal["train", "test"],
        with_y: bool,
        data_source: Literal["train", "test", "both"],
    ):
        """Retrieve data as DataFrame.

        Parameters
        ----------
        dataset : {"train", "test"}
            The dataset to retrieve.

        with_y : bool
            Whether we should check that `y` is not None.

        data_source : {"train", "test", "both"}
            The dataset to analyze. If "train", only the training set is used.
            If "test", only the test set is used. If "both", both sets are concatenated
            vertically.

        Returns
        -------
        X : DataFrame
            The input data.

        y : DataFrame or None
            The target data.
        """
        err_msg = "{} is required when `data_source={!r}`."
        X = getattr(self._parent, f"_X_{dataset}")
        y = getattr(self._parent, f"_y_{dataset}")

        if X is None:
            raise ValueError(err_msg.format(f"X_{dataset}", data_source))
        elif not sbd.is_dataframe(X):
            X = pd.DataFrame(X, columns=[f"Feature {i}" for i in range(X.shape[1])])
        else:
            if not all(isinstance(col, str) for col in X.columns):
                X = X.copy()
                X.columns = [str(col) for col in X.columns]

        if with_y:
            if y is None:
                raise ValueError(err_msg.format(f"y_{dataset}", data_source))

            if isinstance(y, pd.Series) and y.name is not None:
                y = y.to_frame()
            elif not sbd.is_dataframe(y):
                if y.ndim == 1:
                    columns = ["Target"]
                else:
                    columns = [f"Target {i}" for i in range(y.shape[1])]
                y = pd.DataFrame(y, columns=columns)
            else:
                if not all(isinstance(col, str) for col in y.columns):
                    y = y.copy()
                    if y.shape[1] == 1 and list(y.columns) == [0]:
                        y.columns = ["Target"]
                    else:
                        y.columns = [str(col) for col in y.columns]

        return X, y

    def analyze(
        self,
        data_source: Literal["train", "test", "both"] = "both",
        with_y: bool = True,
        subsample: int | None = None,
        subsample_strategy: Literal["head", "random"] = "head",
        seed: int | None = None,
    ) -> TableReportDisplay:
        """Plot dataset statistics.

        Parameters
        ----------
        data_source : {"train", "test", "both"}, default="both"
            The dataset to analyze. If "train", only the training set is used.
            If "test", only the test set is used. If "both", both sets are concatenated
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

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(classifier, **split_data, pos_label=1)
        >>> report.data.analyze().frame()
        """
        if data_source not in (data_source_options := ("train", "test", "both")):
            raise ValueError(
                f"'data_source' options are {data_source_options!r}, got {data_source}."
            )

        if subsample_strategy not in (subsample_strategy_options := ("head", "random")):
            raise ValueError(
                f"'subsample_strategy' options are {subsample_strategy_options!r}, got "
                f"{subsample_strategy}."
            )

        with_y_task_aware = with_y and (self._parent.ml_task != "clustering")

        if data_source == "train":
            X, y = self._retrieve_data_as_frame("train", with_y_task_aware, data_source)
        elif data_source == "test":
            X, y = self._retrieve_data_as_frame("test", with_y_task_aware, data_source)
        else:
            X_train, y_train = self._retrieve_data_as_frame(
                "train", with_y_task_aware, data_source
            )
            X_test, y_test = self._retrieve_data_as_frame(
                "test", with_y_task_aware, data_source
            )
            X = sbd.concat(X_train, X_test, axis=0)
            if with_y_task_aware:
                y = sbd.concat(y_train, y_test, axis=0)

        df = sbd.concat(X, y, axis=1) if with_y_task_aware else X

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
        return self._rich_repr(class_name="skore.EstimatorReport.data")

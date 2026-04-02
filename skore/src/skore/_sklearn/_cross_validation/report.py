from __future__ import annotations

import html
import uuid
from collections.abc import Generator
from typing import TYPE_CHECKING, Literal

import skrub
from joblib import Parallel
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import check_cv
from sklearn.pipeline import Pipeline

from skore._externals._pandas_accessors import DirNamesMixin
from skore._externals._sklearn_compat import _safe_indexing, is_clusterer
from skore._sklearn._base import _BaseReport
from skore._sklearn._diagnostic import check_metrics_consistency_across_folds
from skore._sklearn._estimator.report import EstimatorReport
from skore._sklearn.types import PositiveLabel, SKLearnCrossValidator
from skore._utils._fixes import _validate_joblib_parallel_params
from skore._utils._parallel import delayed
from skore._utils._progress_bar import track
from skore._utils.repr.data import get_documentation_url
from skore._utils.repr.html_repr import render_template

if TYPE_CHECKING:
    from collections.abc import Iterable

    from skore._sklearn._cross_validation.data_accessor import _DataAccessor
    from skore._sklearn._cross_validation.inspection_accessor import (
        _InspectionAccessor,
    )
    from skore._sklearn._cross_validation.metrics_accessor import _MetricsAccessor


def _generate_estimator_report(
    estimator: BaseEstimator,
    X: ArrayLike,
    y: ArrayLike,
    pos_label: PositiveLabel | None,
    train_indices: ArrayLike,
    test_indices: ArrayLike,
) -> EstimatorReport:
    return EstimatorReport(
        estimator,
        fit=True,
        X_train=_safe_indexing(X, train_indices),
        y_train=_safe_indexing(y, train_indices),
        X_test=_safe_indexing(X, test_indices),
        y_test=_safe_indexing(y, test_indices),
        pos_label=pos_label,
    )


class CrossValidationReport(_BaseReport, DirNamesMixin):
    """Report for cross-validation results.

    Upon initialization, `CrossValidationReport` will clone ``estimator`` according to
    ``splitter`` and fit the generated estimators. The fitting is done in parallel.

    Refer to the :ref:`cross_validation_report` section of the user guide for more
    details.

    Parameters
    ----------
    estimator : estimator object
        Estimator to make the cross-validation report from.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target variable to try to predict in the case of supervised learning.

    pos_label : int, float, bool or str, default=None
        For binary classification, the positive class to use for metrics and displays
        that need one. If `None`, skore does not infer a default positive class.
        Binary metrics and displays that support it will expose all classes instead.
        This parameter is rejected for non-binary tasks.

    splitter : int, cross-validation generator or an iterable, default=5
        Determines the cross-validation splitting strategy.
        Possible inputs for `splitter` are:

        - int, to specify the number of splits in a `(Stratified)KFold`,
        - a scikit-learn :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer to scikit-learn's :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        When accessing some methods of the `CrossValidationReport`, the `n_jobs`
        parameter is used to parallelize the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    Attributes
    ----------
    estimator_ : estimator object
        The cloned or copied estimator.

    estimator_name_ : str
        The name of the estimator.

    estimator_reports_ : list of EstimatorReport
        The estimator reports for each split.

    See Also
    --------
    skore.EstimatorReport
        Report for a fitted estimator.

    skore.ComparisonReport
        Report of comparison between estimators.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(random_state=42)
    >>> estimator = LogisticRegression()
    >>> from skore import CrossValidationReport
    >>> report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    """

    _ACCESSOR_CONFIG: dict[str, dict[str, str]] = {
        "data": {"name": "data"},
        "metrics": {"name": "metrics"},
        "inspection": {"name": "inspection"},
    }

    _report_type: Literal["cross-validation"] = "cross-validation"

    metrics: _MetricsAccessor
    inspection: _InspectionAccessor
    data: _DataAccessor

    def __init__(
        self,
        estimator: BaseEstimator,
        X: ArrayLike,
        y: ArrayLike,
        pos_label: PositiveLabel | None = None,
        splitter: int | SKLearnCrossValidator | Generator | None = None,
        n_jobs: int | None = None,
    ) -> None:
        super().__init__()
        if is_clusterer(estimator):
            raise ValueError(
                "Clustering models are not supported yet. Please use a"
                " classification or regression model instead."
            )

        self._estimator = clone(estimator)

        # private storage to ensure properties are read-only
        self._X = X
        self._y = y
        self._pos_label = pos_label
        self._splitter = check_cv(splitter, y, classifier=is_classifier(estimator))
        self._split_indices = tuple(self._splitter.split(self._X, self._y))
        self.n_jobs = n_jobs

        self.estimator_reports_: list[EstimatorReport] = self._fit_estimator_reports()
        self._ml_task = self.estimator_reports_[0].ml_task

    def _fit_estimator_reports(self) -> list[EstimatorReport]:
        """Fit the estimator reports.

        Returns
        -------
        estimator_reports : list of EstimatorReport
            The estimator reports.
        """
        parallel = Parallel(
            **_validate_joblib_parallel_params(
                n_jobs=self.n_jobs, return_as="generator"
            )
        )

        # do not split the data to take advantage of the memory mapping
        return list(
            track(
                parallel(
                    delayed(_generate_estimator_report)(
                        clone(self._estimator),
                        self._X,
                        self._y,
                        self._pos_label,
                        train_indices,
                        test_indices,
                    )
                    for (train_indices, test_indices) in self.split_indices
                ),
                description=f"Processing cross-validation\nfor {self.estimator_name_}",
                total=len(self.split_indices),
            )
        )

    def clear_cache(self) -> None:
        """Clear the cache.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = CrossValidationReport(classifier, X=X, y=y, splitter=2)
        >>> report.cache_predictions()
        >>> report.clear_cache()
        >>> report.estimator_reports_[0]._cache
        {}
        """
        for report in self.estimator_reports_:
            report.clear_cache()

    def cache_predictions(
        self,
        response_methods: list[str] | Literal["auto"] = "auto",
        n_jobs: int | None = None,
    ) -> None:
        """Cache the predictions for sub-estimators reports.

        Parameters
        ----------
        response_methods : {"auto", "predict", "predict_proba", "decision_function"},\
                default="auto"
            The methods to use to compute the predictions.

        n_jobs : int, default=None
            The number of jobs to run in parallel. If `None`, we use the `n_jobs`
            parameter when initializing `CrossValidationReport`.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = CrossValidationReport(classifier, X=X, y=y, splitter=2)
        >>> report.cache_predictions()
        >>> report.estimator_reports_[0]._cache
        {...}
        """
        if n_jobs is None:
            n_jobs = self.n_jobs

        for estimator_report in track(
            self.estimator_reports_,
            description="Cross-validation predictions for split",
        ):
            estimator_report.cache_predictions(
                response_methods=response_methods,
                n_jobs=n_jobs,
            )

    def get_predictions(
        self,
        *,
        data_source: Literal["train", "test"],
        response_method: Literal[
            "predict", "predict_proba", "decision_function"
        ] = "predict",
    ) -> list[ArrayLike]:
        """Get estimator's predictions.

        This method has the advantage to reload from the cache if the predictions
        were already computed in a previous call.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        response_method : {"predict", "predict_proba", "decision_function"}, \
                default="predict"
            The response method to use to get the predictions.

        Returns
        -------
        list of np.ndarray of shape (n_samples,) or (n_samples, n_classes)
            The predictions for each cross-validation split.

        Raises
        ------
        ValueError
            If the data source is invalid.

        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.linear_model import LogisticRegression
        >>> X, y = make_classification(random_state=42)
        >>> estimator = LogisticRegression()
        >>> from skore import CrossValidationReport
        >>> report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
        >>> predictions = report.get_predictions(data_source="test")
        >>> print([split_predictions.shape for split_predictions in predictions])
        [(50,), (50,)]
        """
        if data_source not in ("train", "test"):
            raise ValueError(
                f"Invalid data source: {data_source}. Valid data sources are "
                "'train' and 'test'."
            )
        return [
            report.get_predictions(
                data_source=data_source,
                response_method=response_method,
            )
            for report in self.estimator_reports_
        ]

    def create_estimator_report(
        self, *, X_test: ArrayLike, y_test: ArrayLike
    ) -> EstimatorReport:
        """Create an estimator report from the cross-validation report.

        This method creates a new :class:`~skore.EstimatorReport` with the same
        estimator and the same data as the cross-validation report. It is useful to
        evaluate and deploy a model that was deemed optimal with cross-validation.
        Provide a held out test set to properly evaluate the performance of the model.

        Parameters
        ----------
        X_test : {array-like, sparse matrix} of shape (n_samples, n_features)
            Testing data. It should have the same structure as the training data.

        y_test : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Testing target.

        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import ComparisonReport, CrossValidationReport
        >>> X, y = make_classification(random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        >>> linear_report = CrossValidationReport(
        ...     LogisticRegression(random_state=42), X_train, y_train
        ... )
        >>> forest_report = CrossValidationReport(
        ...     RandomForestClassifier(random_state=42), X_train, y_train
        ... )
        >>> comparison_report = ComparisonReport([linear_report, forest_report])
        >>> summary = comparison_report.metrics.summarize().frame()

        >>> # Notice that e.g. the RandomForestClassifier performs best
        >>> final_report = forest_report.create_estimator_report(
        ...     X_test=X_test, y_test=y_test
        ... )
        >>> final_report.metrics.summarize().frame()

        Returns
        -------
        report : :class:`~skore.EstimatorReport`
            The estimator report.
        """
        report = EstimatorReport(
            self._estimator,
            X_train=self._X,
            y_train=self._y,
            X_test=X_test,
            y_test=y_test,
            pos_label=self._pos_label,
        )
        return report

    def _run_checks(
        self,
    ) -> tuple[dict[str, dict], set[str]]:
        total_splits = len(self.estimator_reports_)
        all_checked_codes: set[str] = set()
        positives_by_code: dict[str, list[dict]] = {}

        for estimator_report in self.estimator_reports_:
            results, checked_codes = estimator_report._get_issues()
            all_checked_codes |= checked_codes
            for code, diagnostic in results.items():
                positives_by_code.setdefault(code, []).append(diagnostic)

        issues: dict[str, dict] = {}
        for code in all_checked_codes:
            positives = positives_by_code.get(code, [])
            if len(positives) > total_splits / 2:
                ref = positives[0]
                issues[code] = {
                    "title": ref["title"],
                    "docs_anchor": ref["docs_anchor"],
                    "explanation": (
                        f"Detected in {len(positives)}/{total_splits} evaluated splits."
                    ),
                }

        issues.update(check_metrics_consistency_across_folds(self))
        all_checked_codes |= {"SKD003"}
        return issues, all_checked_codes

    @property
    def ml_task(self) -> str:
        return self._ml_task

    @property
    def estimator(self) -> BaseEstimator:
        return self._estimator

    @property
    def estimator_(self) -> BaseEstimator:
        return self._estimator

    @property
    def estimator_name_(self) -> str:
        if isinstance(self._estimator, Pipeline):
            name = self._estimator[-1].__class__.__name__
        else:
            name = self._estimator.__class__.__name__
        return name

    @property
    def X(self) -> ArrayLike:
        return self._X

    @property
    def y(self) -> ArrayLike | None:
        return self._y

    @property
    def splitter(self) -> SKLearnCrossValidator:
        return self._splitter

    @property
    def split_indices(self) -> tuple[tuple[Iterable[int], Iterable[int]]]:
        return self._split_indices

    @property
    def pos_label(self) -> PositiveLabel | None:
        return self._pos_label

    ####################################################################################
    # Methods related to the help and repr
    ####################################################################################

    def _get_help_title(self) -> str:
        return f"Tools to diagnose estimator {self.estimator_name_}"

    def _get_help_legend(self) -> str:
        return (
            "[cyan](↗︎)[/cyan] higher is better [orange1](↘︎)[/orange1] lower is better"
        )

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"{self.__class__.__name__}(estimator={self.estimator_}, ...)"

    def _html_repr_fragments(self) -> dict[str, str]:
        """HTML snippets for the report body (metrics, estimator diagram, data table).

        Used by :meth:`_repr_html_` and by :class:`~skore.ComparisonReport` to embed
        one report's views in the comparison HTML repr.
        """
        metrics_html = (
            self.metrics.summarize(data_source="test")
            .frame(aggregate=("mean", "std"), favorability=False)
            ._repr_html_()
        )

        df = self.data._prepare_dataframe_for_display(
            with_y=self.ml_task != "clustering"
        )
        table_report = skrub.TableReport(
            df,
            max_plot_columns=0,
            max_association_columns=0,
            verbose=False,
        )
        table_report._set_minimal_mode()
        table_report_html = table_report.html_snippet()

        try:
            estimator_html = self.estimator_._repr_html_()
        except Exception:
            estimator_html = f"<p>{html.escape(repr(self.estimator_))}</p>"

        issues, checked_codes = self._get_issues()
        diagnostic_html = (
            f"<div class='report-diagnostic-details'>{len(issues)} "
            f"issue(s) detected, {len(checked_codes)} check(s) ran.</div>"
        )

        return {
            "metrics_summary": metrics_html,
            "estimator_display": estimator_html,
            "table_report": table_report_html,
            "diagnostic": diagnostic_html,
        }

    def _repr_html_(self) -> str:
        """HTML representation of the cross-validation report."""
        fragments = self._html_repr_fragments()
        container_id = f"skore-cross-validation-report-{uuid.uuid4().hex[:8]}"
        help_doc_url = get_documentation_url(obj=self, method_name="help")
        report_class_name = self.__class__.__name__
        metrics_accessor_doc_url = get_documentation_url(
            obj=self, accessor_name="metrics"
        )
        inspection_accessor_doc_url = get_documentation_url(
            obj=self, accessor_name="inspection"
        )
        data_accessor_doc_url = get_documentation_url(obj=self, accessor_name="data")
        diagnose_documentation_url = get_documentation_url(
            obj=self, method_name="diagnose"
        )
        return render_template(
            "cross_validation_report.html.j2",
            {
                "container_id": container_id,
                "help_doc_url": help_doc_url,
                "report_class_name": report_class_name,
                "metrics_accessor_doc_url": metrics_accessor_doc_url,
                "inspection_accessor_doc_url": inspection_accessor_doc_url,
                "data_accessor_doc_url": data_accessor_doc_url,
                "diagnose_documentation_url": diagnose_documentation_url,
                **fragments,
            },
        )

    def _repr_mimebundle_(self, **kwargs):
        """Mime bundle used by Jupyter kernels to display the report."""
        output = {"text/plain": repr(self)}
        output["text/html"] = self._repr_html_()
        return output

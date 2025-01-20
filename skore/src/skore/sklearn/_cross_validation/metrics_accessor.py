import joblib
import numpy as np
import pandas as pd
from sklearn.utils.metaestimators import available_if

from skore.externals._pandas_accessors import DirNamesMixin
from skore.sklearn._base import _BaseAccessor, _get_cached_response_values
from skore.sklearn._plot import (
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
)
from skore.utils._accessor import _check_supported_ml_task
from skore.utils._progress_bar import progress_decorator

###############################################################################
# Metrics accessor
###############################################################################


class _MetricsAccessor(_BaseAccessor, DirNamesMixin):
    """Accessor for metrics-related operations.

    You can access this accessor using the `metrics` attribute.
    """

    _SCORE_OR_LOSS_ICONS = {
        "accuracy": "(↗︎)",
        "precision": "(↗︎)",
        "recall": "(↗︎)",
        "brier_score": "(↘︎)",
        "roc_auc": "(↗︎)",
        "log_loss": "(↘︎)",
        "r2": "(↗︎)",
        "rmse": "(↘︎)",
        "report_metrics": "",
        "custom_metric": "",
    }

    def __init__(self, parent):
        super().__init__(parent)

        self._parent_progress = None

    def report_metrics(
        self,
        *,
        data_source="test",
        scoring=None,
        scoring_names=None,
        pos_label=None,
        scoring_kwargs=None,
        aggregate=None,
    ):
        """Report a set of metrics for our estimator.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.

        scoring : list of str, callable, or scorer, default=None
            The metrics to report. You can get the possible list of string by calling
            `reporter.metrics.help()`. When passing a callable, it should take as
            arguments `y_true`, `y_pred` as the two first arguments. Additional
            arguments can be passed as keyword arguments and will be forwarded with
            `scoring_kwargs`. If the callable API is too restrictive (e.g. need to pass
            same parameter name with different values), you can use scikit-learn scorers
            as provided by :func:`sklearn.metrics.make_scorer`.

        scoring_names : list of str, default=None
            Used to overwrite the default scoring names in the report. It should be of
            the same length as the `scoring` parameter.

        pos_label : int, float, bool or str, default=None
            The positive class.

        scoring_kwargs : dict, default=None
            The keyword arguments to pass to the scoring functions.

        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the scores across the cross-validation splits.

        Returns
        -------
        pd.DataFrame
            The statistics for the metrics.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> reporter = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
        Processing cross-validation ...
        >>> reporter.metrics.report_metrics(
        ...     scoring=["precision", "recall"], pos_label=1, aggregate=["mean", "std"]
        ... )
        Compute metric for each split ...
        Metric                   Precision (↗︎)  Recall (↗︎)
        LogisticRegression mean        0.94...     0.96...
                           std         0.02...     0.02...
        """
        return self._compute_metric_scores(
            report_metric_name="report_metrics",
            data_source=data_source,
            aggregate=aggregate,
            scoring=scoring,
            pos_label=pos_label,
            scoring_kwargs=scoring_kwargs,
            scoring_names=scoring_names,
        )

    @progress_decorator(description="Compute metric for each split")
    def _compute_metric_scores(
        self,
        report_metric_name,
        *,
        data_source="test",
        aggregate=None,
        **metric_kwargs,
    ):
        cache_key = (self._parent._hash, report_metric_name, data_source)
        cache_key += (aggregate,) if aggregate is None else tuple(aggregate)

        if metric_kwargs != {}:
            # we need to enforce the order of the parameter for a specific metric
            # to make sure that we hit the cache in a consistent way
            ordered_metric_kwargs = sorted(metric_kwargs.keys())

            for key in ordered_metric_kwargs:
                if isinstance(metric_kwargs[key], (np.ndarray, list, dict)):
                    cache_key += (joblib.hash(metric_kwargs[key]),)
                else:
                    cache_key += (metric_kwargs[key],)

        progress = self._progress_info["current_progress"]
        main_task = self._progress_info["current_task"]

        total_estimators = len(self._parent.estimator_reports_)
        progress.update(main_task, total=total_estimators)

        if cache_key in self._parent._cache:
            results = self._parent._cache[cache_key]
        else:
            parallel = joblib.Parallel(
                n_jobs=self._parent.n_jobs,
                return_as="generator",
                require="sharedmem",
            )
            generator = parallel(
                joblib.delayed(getattr(report.metrics, report_metric_name))(
                    data_source=data_source, **metric_kwargs
                )
                for report in self._parent.estimator_reports_
            )
            results = []
            for result in generator:
                results.append(result)
                progress.update(main_task, advance=1, refresh=True)

            results = pd.concat(
                results,
                axis=0,
                keys=[f"Split #{i}" for i in range(len(results))],
            )
            results = results.swaplevel(0, 1)
            if aggregate:
                if isinstance(aggregate, str):
                    aggregate = [aggregate]
                results = results.aggregate(func=aggregate, axis=0)
                results = pd.concat([results], keys=[self._parent.estimator_name_])

            self._parent._cache[cache_key] = results
        return results

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def accuracy(self, *, data_source="test", aggregate=None):
        """Compute the accuracy score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.

        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the scores across the cross-validation splits.

        Returns
        -------
        pd.DataFrame
            The accuracy score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> reporter = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
        Processing cross-validation ...
        >>> reporter.metrics.accuracy()
        Compute metric for each split ...
        Metric                       Accuracy (↗︎)
        LogisticRegression Split #0       0.94...
                           Split #1       0.94...
        """
        return self._compute_metric_scores(
            report_metric_name="accuracy",
            data_source=data_source,
            aggregate=aggregate,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def precision(
        self,
        *,
        data_source="test",
        average=None,
        pos_label=None,
        aggregate=None,
    ):
        """Compute the precision score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.

        average : {"binary","macro", "micro", "weighted", "samples"} or None, \
                default=None
            Used with multiclass problems.
            If `None`, the metrics for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:

            - "binary": Only report results for the class specified by `pos_label`.
              This is applicable only if targets (`y_{true,pred}`) are binary.
            - "micro": Calculate metrics globally by counting the total true positives,
              false negatives and false positives.
            - "macro": Calculate metrics for each label, and find their unweighted
              mean.  This does not take label imbalance into account.
            - "weighted": Calculate metrics for each label, and find their average
              weighted by support (the number of true instances for each label). This
              alters 'macro' to account for label imbalance; it can result in an F-score
              that is not between precision and recall.
            - "samples": Calculate metrics for each instance, and find their average
              (only meaningful for multilabel classification where this differs from
              :func:`accuracy_score`).

            .. note::
                If `pos_label` is specified and `average` is None, then we report
                only the statistics of the positive class (i.e. equivalent to
                `average="binary"`).

        pos_label : int, float, bool or str, default=None
            The positive class.

        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the scores across the cross-validation splits.

        Returns
        -------
        pd.DataFrame
            The precision score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> reporter = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
        Processing cross-validation ...
        >>> reporter.metrics.precision()
        Compute metric for each split ...
        Metric                      Precision (↗︎)
        Class label                              0         1
        LogisticRegression Split #0       0.96...   0.93...
                           Split #1       0.90...   0.96...
        """
        return self._compute_metric_scores(
            report_metric_name="precision",
            data_source=data_source,
            aggregate=aggregate,
            average=average,
            pos_label=pos_label,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def recall(
        self,
        *,
        data_source="test",
        average=None,
        pos_label=None,
        aggregate=None,
    ):
        """Compute the recall score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.

        average : {"binary","macro", "micro", "weighted", "samples"} or None, \
                default=None
            Used with multiclass problems.
            If `None`, the metrics for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:

            - "binary": Only report results for the class specified by `pos_label`.
              This is applicable only if targets (`y_{true,pred}`) are binary.
            - "micro": Calculate metrics globally by counting the total true positives,
              false negatives and false positives.
            - "macro": Calculate metrics for each label, and find their unweighted
              mean.  This does not take label imbalance into account.
            - "weighted": Calculate metrics for each label, and find their average
              weighted by support (the number of true instances for each label). This
              alters 'macro' to account for label imbalance; it can result in an F-score
              that is not between precision and recall. Weighted recall is equal to
              accuracy.
            - "samples": Calculate metrics for each instance, and find their average
              (only meaningful for multilabel classification where this differs from
              :func:`accuracy_score`).

            .. note::
                If `pos_label` is specified and `average` is None, then we report
                only the statistics of the positive class (i.e. equivalent to
                `average="binary"`).

        pos_label : int, float, bool or str, default=None
            The positive class.

        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the scores across the cross-validation splits.

        Returns
        -------
        pd.DataFrame
            The recall score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> reporter = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
        Processing cross-validation ...
        >>> reporter.metrics.recall()
        Compute metric for each split ...
        Metric                      Recall (↗︎)
        Class label                           0        1
        LogisticRegression Split #0    0.87...   0.98...
                           Split #1    0.94...   0.94...
        """
        return self._compute_metric_scores(
            report_metric_name="recall",
            data_source=data_source,
            aggregate=aggregate,
            average=average,
            pos_label=pos_label,
        )

    @available_if(
        _check_supported_ml_task(supported_ml_tasks=["binary-classification"])
    )
    def brier_score(self, *, data_source="test", aggregate=None):
        """Compute the Brier score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.

        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the scores across the cross-validation splits.

        Returns
        -------
        pd.DataFrame
            The Brier score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> reporter = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
        Processing cross-validation ...
        >>> reporter.metrics.brier_score()
        Compute metric for each split ...
        Metric                       Brier score (↘︎)
        LogisticRegression Split #0          0.04...
                           Split #1          0.04...
        """
        return self._compute_metric_scores(
            report_metric_name="brier_score",
            data_source=data_source,
            aggregate=aggregate,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def roc_auc(
        self,
        *,
        data_source="test",
        average=None,
        multi_class="ovr",
        aggregate=None,
    ):
        """Compute the ROC AUC score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.

        average : {"auto", "macro", "micro", "weighted", "samples"}, \
                default=None
            Average to compute the ROC AUC score in a multiclass setting. By default,
            no average is computed. Otherwise, this determines the type of averaging
            performed on the data.

            - "micro": Calculate metrics globally by considering each element of
              the label indicator matrix as a label.
            - "macro": Calculate metrics for each label, and find their unweighted
              mean. This does not take label imbalance into account.
            - "weighted": Calculate metrics for each label, and find their average,
              weighted by support (the number of true instances for each label).
            - "samples": Calculate metrics for each instance, and find their
              average.

            .. note::
                Multiclass ROC AUC currently only handles the "macro" and
                "weighted" averages. For multiclass targets, `average=None` is only
                implemented for `multi_class="ovr"` and `average="micro"` is only
                implemented for `multi_class="ovr"`.

        multi_class : {"raise", "ovr", "ovo"}, default="ovr"
            The multi-class strategy to use.

            - "raise": Raise an error if the data is multiclass.
            - "ovr": Stands for One-vs-rest. Computes the AUC of each class against the
              rest. This treats the multiclass case in the same way as the multilabel
              case. Sensitive to class imbalance even when `average == "macro"`,
              because class imbalance affects the composition of each of the "rest"
              groupings.
            - "ovo": Stands for One-vs-one. Computes the average AUC of all possible
              pairwise combinations of classes. Insensitive to class imbalance when
              `average == "macro"`.

        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the scores across the cross-validation splits.

        Returns
        -------
        pd.DataFrame
            The ROC AUC score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> reporter = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
        Processing cross-validation ...
        >>> reporter.metrics.roc_auc()
        Compute metric for each split ...
        Metric                       ROC AUC (↗︎)
        LogisticRegression Split #0      0.99...
                           Split #1      0.98...
        """
        return self._compute_metric_scores(
            report_metric_name="roc_auc",
            data_source=data_source,
            aggregate=aggregate,
            average=average,
            multi_class=multi_class,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def log_loss(self, *, data_source="test", aggregate=None):
        """Compute the log loss.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.

        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the scores across the cross-validation splits.

        Returns
        -------
        pd.DataFrame
            The log-loss.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> reporter = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
        Processing cross-validation ...
        >>> reporter.metrics.log_loss()
        Compute metric for each split ...
        Metric                       Log loss (↘︎)
        LogisticRegression Split #0       0.1...
                           Split #1       0.1...
        """
        return self._compute_metric_scores(
            report_metric_name="log_loss",
            data_source=data_source,
            aggregate=aggregate,
        )

    @available_if(_check_supported_ml_task(supported_ml_tasks=["regression"]))
    def r2(
        self,
        *,
        data_source="test",
        multioutput="raw_values",
        aggregate=None,
    ):
        """Compute the R² score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.

        multioutput : {"raw_values", "uniform_average"} or array-like of shape \
                (n_outputs,), default="raw_values"
            Defines aggregating of multiple output values. Array-like value defines
            weights used to average errors. The other possible values are:

            - "raw_values": Returns a full set of errors in case of multioutput input.
            - "uniform_average": Errors of all outputs are averaged with uniform weight.

            By default, no averaging is done.

        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the scores across the cross-validation splits.

        Returns
        -------
        pd.DataFrame
            The R² score.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import CrossValidationReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> regressor = Ridge()
        >>> reporter = CrossValidationReport(regressor, X=X, y=y, cv_splitter=2)
        Processing cross-validation ...
        >>> reporter.metrics.r2()
        Compute metric for each split ...
        Metric           R² (↗︎)
        Ridge Split #0  0.36...
              Split #1  0.39...
        """
        return self._compute_metric_scores(
            report_metric_name="r2",
            data_source=data_source,
            aggregate=aggregate,
            multioutput=multioutput,
        )

    @available_if(_check_supported_ml_task(supported_ml_tasks=["regression"]))
    def rmse(
        self,
        *,
        data_source="test",
        multioutput="raw_values",
        aggregate=None,
    ):
        """Compute the root mean squared error.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.

        multioutput : {"raw_values", "uniform_average"} or array-like of shape \
                (n_outputs,), default="raw_values"
            Defines aggregating of multiple output values. Array-like value defines
            weights used to average errors. The other possible values are:

            - "raw_values": Returns a full set of errors in case of multioutput input.
            - "uniform_average": Errors of all outputs are averaged with uniform weight.

            By default, no averaging is done.

        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the scores across the cross-validation splits.

        Returns
        -------
        pd.DataFrame
            The root mean squared error.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import CrossValidationReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> regressor = Ridge()
        >>> reporter = CrossValidationReport(regressor, X=X, y=y, cv_splitter=2)
        Processing cross-validation ...
        >>> reporter.metrics.rmse()
        Compute metric for each split ...
        Metric          RMSE (↘︎)
        Ridge Split #0  59.9...
              Split #1  61.4...
        """
        return self._compute_metric_scores(
            report_metric_name="rmse",
            data_source=data_source,
            aggregate=aggregate,
            multioutput=multioutput,
        )

    def custom_metric(
        self,
        metric_function,
        response_method,
        *,
        metric_name=None,
        data_source="test",
        aggregate=None,
        **kwargs,
    ):
        """Compute a custom metric.

        It brings some flexibility to compute any desired metric. However, we need to
        follow some rules:

        - `metric_function` should take `y_true` and `y_pred` as the first two
          positional arguments.
        - `response_method` corresponds to the estimator's method to be invoked to get
          the predictions. It can be a string or a list of strings to defined in which
          order the methods should be invoked.

        Parameters
        ----------
        metric_function : callable
            The metric function to be computed. The expected signature is
            `metric_function(y_true, y_pred, **kwargs)`.

        response_method : str or list of str
            The estimator's method to be invoked to get the predictions. The possible
            values are: `predict`, `predict_proba`, `predict_log_proba`, and
            `decision_function`.

        metric_name : str, default=None
            The name of the metric. If not provided, it will be inferred from the
            metric function.

        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.

        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the scores across the cross-validation splits.

        **kwargs : dict
            Any additional keyword arguments to be passed to the metric function.

        Returns
        -------
        pd.DataFrame
            The custom metric.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from sklearn.metrics import mean_absolute_error
        >>> from skore import CrossValidationReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> regressor = Ridge()
        >>> reporter = CrossValidationReport(regressor, X=X, y=y, cv_splitter=2)
        Processing cross-validation ...
        >>> reporter.metrics.custom_metric(
        ...     metric_function=mean_absolute_error,
        ...     response_method="predict",
        ...     metric_name="MAE (↗︎)",
        ... )
        Compute metric for each split ...
        Metric           MAE (↗︎)
        Ridge Split #0  50.1...
              Split #1  52.6...
        """
        return self._compute_metric_scores(
            report_metric_name="custom_metric",
            data_source=data_source,
            aggregate=aggregate,
            metric_function=metric_function,
            response_method=response_method,
            metric_name=metric_name,
            **kwargs,
        )

    ####################################################################################
    # Methods related to the help tree
    ####################################################################################

    def _sort_methods_for_help(self, methods):
        """Override sort method for metrics-specific ordering.

        In short, we display the `report_metrics` first and then the `custom_metric`.
        """

        def _sort_key(method):
            name = method[0]
            if name == "custom_metric":
                priority = 1
            elif name == "report_metrics":
                priority = 2
            else:
                priority = 0
            return priority, name

        return sorted(methods, key=_sort_key)

    def _format_method_name(self, name):
        """Override format method for metrics-specific naming."""
        method_name = f"{name}(...)"
        method_name = method_name.ljust(22)
        if self._SCORE_OR_LOSS_ICONS[name] in ("(↗︎)", "(↘︎)"):
            if self._SCORE_OR_LOSS_ICONS[name] == "(↗︎)":
                method_name += f"[cyan]{self._SCORE_OR_LOSS_ICONS[name]}[/cyan]"
                return method_name.ljust(43)
            else:  # (↘︎)
                method_name += f"[orange1]{self._SCORE_OR_LOSS_ICONS[name]}[/orange1]"
                return method_name.ljust(49)
        else:
            return method_name.ljust(29)

    def _get_methods_for_help(self):
        """Override to exclude the plot accessor from methods list."""
        methods = super()._get_methods_for_help()
        return [(name, method) for name, method in methods if name != "plot"]

    def _create_help_tree(self):
        """Override to include plot methods in a separate branch."""
        tree = super()._create_help_tree()

        # Add plot methods in a separate branch
        plot_branch = tree.add("[bold cyan].plot :art:[/bold cyan]")
        plot_methods = self.plot._get_methods_for_help()
        plot_methods = self.plot._sort_methods_for_help(plot_methods)

        for name, method in plot_methods:
            displayed_name = self.plot._format_method_name(name)
            description = self.plot._get_method_description(method)
            plot_branch.add(f".{displayed_name}".ljust(27) + f"- {description}")

        return tree

    def _get_help_panel_title(self):
        return "[bold cyan]Available metrics methods[/bold cyan]"

    def _get_help_legend(self):
        return (
            "[cyan](↗︎)[/cyan] higher is better [orange1](↘︎)[/orange1] lower is better"
        )

    def _get_help_tree_title(self):
        return "[bold cyan]reporter.metrics[/bold cyan]"

    def __repr__(self):
        """Return a string representation using rich."""
        return self._rich_repr(
            class_name="skore.CrossValidationReport.metrics",
            help_method_name="reporter.metrics.help()",
        )


########################################################################################
# Sub-accessors
# Plotting
########################################################################################


class _PlotMetricsAccessor(_BaseAccessor):
    """Plotting methods for the metrics accessor."""

    def __init__(self, parent):
        super().__init__(parent._parent)
        self._metrics_parent = parent
        self._parent_progress = None

    @progress_decorator(description="Computing predictions for display")
    def _get_display(
        self,
        *,
        data_source,
        response_method,
        display_class,
        display_kwargs,
        display_plot_kwargs,
    ):
        """Get the display from the cache or compute it.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.

        response_method : str
            The response method.

        display_class : class
            The display class.

        display_kwargs : dict
            The display kwargs used by `display_class._from_predictions`.

        display_plot_kwargs : dict
            The display kwargs used by `display.plot`.

        Returns
        -------
        display : display_class
            The display.
        """
        cache_key = (self._parent._hash, display_class.__name__)
        cache_key += tuple(display_kwargs.values())
        cache_key += (data_source,)

        progress = self._progress_info["current_progress"]
        main_task = self._progress_info["current_task"]
        total_estimators = len(self._parent.estimator_reports_)
        progress.update(main_task, total=total_estimators)

        if cache_key in self._parent._cache:
            display = self._parent._cache[cache_key]
            display.plot(**display_plot_kwargs)
        else:
            y_true, y_pred = [], []
            for report in self._parent.estimator_reports_:
                X, y, _ = report.metrics._get_X_y_and_data_source_hash(
                    data_source=data_source
                )
                y_true.append(y)
                y_pred.append(
                    _get_cached_response_values(
                        cache=report._cache,
                        estimator_hash=report._hash,
                        estimator=report._estimator,
                        X=X,
                        response_method=response_method,
                        data_source=data_source,
                        data_source_hash=None,
                        pos_label=display_kwargs.get("pos_label", None),
                    )
                )
                progress.update(main_task, advance=1, refresh=True)

            display = display_class._from_predictions(
                y_true,
                y_pred,
                estimator=self._parent.estimator_reports_[0]._estimator,
                estimator_name=self._parent.estimator_name_,
                ml_task=self._parent._ml_task,
                data_source=data_source,
                **display_kwargs,
                **display_plot_kwargs,
            )
            self._parent._cache[cache_key] = display

        return display

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def roc(self, *, data_source="test", pos_label=None, ax=None):
        """Plot the ROC curve.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.

        pos_label : int, float, bool or str, default=None
            The positive class.

        ax : matplotlib.axes.Axes, default=None
            The axes to plot on.

        Returns
        -------
        RocCurveDisplay
            The ROC curve display.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> reporter = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
        Processing cross-validation ...
        >>> display = reporter.metrics.plot.roc()
        Computing predictions for display ...
        >>> display.plot(roc_curve_kwargs={"color": "tab:red"})
        """
        response_method = ("predict_proba", "decision_function")
        display_kwargs = {"pos_label": pos_label}
        display_plot_kwargs = {"ax": ax, "plot_chance_level": True, "despine": True}
        return self._get_display(
            data_source=data_source,
            response_method=response_method,
            display_class=RocCurveDisplay,
            display_kwargs=display_kwargs,
            display_plot_kwargs=display_plot_kwargs,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def precision_recall(self, *, data_source="test", pos_label=None, ax=None):
        """Plot the precision-recall curve.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.

        pos_label : int, float, bool or str, default=None
            The positive class.

        ax : matplotlib.axes.Axes, default=None
            The axes to plot on.

        Returns
        -------
        PrecisionRecallCurveDisplay
            The precision-recall curve display.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> reporter = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
        Processing cross-validation ...
        >>> display = reporter.metrics.plot.precision_recall()
        Computing predictions for display ...
        >>> display.plot()
        """
        response_method = ("predict_proba", "decision_function")
        display_kwargs = {"pos_label": pos_label}
        display_plot_kwargs = {"ax": ax, "despine": True}
        return self._get_display(
            data_source=data_source,
            response_method=response_method,
            display_class=PrecisionRecallCurveDisplay,
            display_kwargs=display_kwargs,
            display_plot_kwargs=display_plot_kwargs,
        )

    @available_if(_check_supported_ml_task(supported_ml_tasks=["regression"]))
    def prediction_error(
        self,
        *,
        data_source="test",
        ax=None,
        kind="residual_vs_predicted",
        subsample=1_000,
        random_state=None,
    ):
        """Plot the prediction error of a regression model.

        Extra keyword arguments will be passed to matplotlib's `plot`.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        kind : {"actual_vs_predicted", "residual_vs_predicted"}, \
                default="residual_vs_predicted"
            The type of plot to draw:

            - "actual_vs_predicted" draws the observed values (y-axis) vs.
              the predicted values (x-axis).
            - "residual_vs_predicted" draws the residuals, i.e. difference
              between observed and predicted values, (y-axis) vs. the predicted
              values (x-axis).

        subsample : float, int or None, default=1_000
            Sampling the samples to be shown on the scatter plot. If `float`,
            it should be between 0 and 1 and represents the proportion of the
            original dataset. If `int`, it represents the number of samples
            display on the scatter plot. If `None`, no subsampling will be
            applied. by default, 1,000 samples or less will be displayed.

        random_state : int, default=None
            The random state to use for the subsampling.

        Returns
        -------
        PredictionErrorDisplay
            The prediction error display.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import CrossValidationReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> regressor = Ridge()
        >>> reporter = CrossValidationReport(regressor, X=X, y=y, cv_splitter=2)
        Processing cross-validation ...
        >>> display = reporter.metrics.plot.prediction_error(
        ...     kind="actual_vs_predicted"
        ... )
        Computing predictions for display ...
        >>> display.plot(line_kwargs={"color": "tab:red"})
        """
        display_kwargs = {"subsample": subsample, "random_state": random_state}
        display_plot_kwargs = {"ax": ax, "kind": kind}
        return self._get_display(
            data_source=data_source,
            response_method="predict",
            display_class=PredictionErrorDisplay,
            display_kwargs=display_kwargs,
            display_plot_kwargs=display_plot_kwargs,
        )

    def _get_help_panel_title(self):
        return "[bold cyan]Available plot methods[/bold cyan]"

    def _get_help_tree_title(self):
        return "[bold cyan]reporter.metrics.plot[/bold cyan]"

    def __repr__(self):
        """Return a string representation using rich."""
        return self._rich_repr(
            class_name="skore.CrossValidationReport.metrics.plot",
            help_method_name="reporter.metrics.plot.help()",
        )

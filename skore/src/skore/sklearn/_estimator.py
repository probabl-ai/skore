import inspect

from rich.tree import Tree
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.utils._response import _check_response_method, _get_response_values
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from skore.sklearn.find_ml_task import _find_ml_task
from skore.utils._accessor import _check_supported_ml_task, register_accessor


def _check_supported_estimator(supported_estimators):
    def check(accessor):
        estimator = accessor._parent.estimator
        if isinstance(estimator, Pipeline):
            estimator = estimator.steps[-1][1]
        supported_estimator = isinstance(estimator, supported_estimators)

        if not supported_estimator:
            raise AttributeError(
                f"The {estimator.__class__.__name__} estimator is not supported "
                "by the function called."
            )

        return True

    return check


class EstimatorReport:
    def __init__(self, estimator, *, X_train, y_train, X_val, y_val):
        check_is_fitted(estimator)

        self.estimator = estimator
        if isinstance(estimator, Pipeline):
            self.estimator_name = self.estimator[-1].__class__.__name__
        else:
            self.estimator_name = self.estimator.__class__.__name__

        self.X_train = X_train
        self.y_train = y_train

        self.X_val = X_val
        self.y_val = y_val

        self._cache = {}  # in-memory cache for the report
        self._ml_task = _find_ml_task(self.y_val, estimator=self.estimator)

    @classmethod
    def from_fitted_estimator(cls, estimator, *, X, y=None):
        return cls(estimator=estimator, X_train=None, y_train=None, X_val=X, y_val=y)

    @classmethod
    def from_unfitted_estimator(
        cls, estimator, *, X_train, y_train=None, X_test=None, y_test=None
    ):
        estimator = clone(estimator).fit(X_train, y_train)
        return cls(
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
        )

    def _get_cached_response_values(
        self,
        *,
        hash,
        estimator,
        X,
        response_method,
        pos_label=None,
    ):
        prediction_method = _check_response_method(estimator, response_method).__name__
        if prediction_method in ("predict_proba", "decision_function"):
            # pos_label is only important in classification and with probabilities
            # and decision functions
            cache_key = (hash, pos_label, prediction_method)
        else:
            cache_key = (hash, prediction_method)

        if cache_key in self._cache:
            return self._cache[cache_key]

        predictions, _ = _get_response_values(
            estimator,
            X=X,
            response_method=prediction_method,
            pos_label=pos_label,
            return_response_method_used=False,
        )
        self._cache[cache_key] = predictions

        return predictions

    def help(self):
        """Display available plotting and metrics functions using rich."""
        from skore import console  # avoid circular import

        tree = Tree(f"🔧 Available tools for {self.estimator_name} estimator")

        def _add_accessor_methods_to_tree(tree, accessor, icon, accessor_name):
            branch = tree.add(f"{icon} {accessor_name}")
            methods = inspect.getmembers(accessor, predicate=inspect.ismethod)
            for name, method in methods:
                if not name.startswith("_") and not name.startswith("__"):
                    doc = (
                        method.__doc__.split("\n")[0]
                        if method.__doc__
                        else "No description available"
                    )
                    branch.add(f"[green]{name}[/green] - {doc}")

        _add_accessor_methods_to_tree(tree, self.plot, "🎨", "plot")
        # _add_accessor_methods_to_tree(tree, self.metrics, "📏", "metrics")
        # _add_accessor_methods_to_tree(tree, self.inspection, "🔍", "inspection")
        # _add_accessor_methods_to_tree(tree, self.linting, "✔️", "linting")

        console.print(tree)


@register_accessor("plot", EstimatorReport)
class _PlotAccessor:
    def __init__(self, parent):
        self._parent = parent

    @available_if(
        _check_supported_ml_task(supported_ml_tasks=["binary-classification"])
    )
    def roc(self):
        pass

"""CrossValidationItem class.

This class represents the output of a cross-validation workflow.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import json
import re
import statistics
from functools import cached_property
from typing import TYPE_CHECKING, Any, Union

import numpy
import plotly.graph_objects
import plotly.io

from skore.item.item import Item, ItemTypeError
from skore.sklearn.cross_validation import CrossValidationReporter

if TYPE_CHECKING:
    CVSplitter = Any


def _hash_numpy(arr: numpy.ndarray) -> str:
    """Compute a hash string from a numpy array.

    Parameters
    ----------
    arr : numpy array
        The numpy array whose hash will be computed.

    Returns
    -------
    hash : str
        A hash corresponding to the input array.
    """
    return hashlib.sha256(bytes(memoryview(arr))).hexdigest()


def _metric_title(metric):
    m = metric.replace("_", " ")
    title = f"Mean {m}"
    if title.endswith(" time"):
        title = title + " (seconds)"
    return title


# Data used for training, passed as input to scikit-learn
Data = Any
# Target used for training, passed as input to scikit-learn
Target = Any


class CrossValidationItem(Item):
    """
    A class to represent the output of a cross-validation workflow.

    This class encapsulates the output of the
    :func:`sklearn.model_selection.cross_validate` function along with its creation and
    update timestamps.
    """

    def __init__(
        self,
        cv_results_serialized: dict,
        estimator_info: dict,
        X_info: dict,
        y_info: Union[dict, None],
        plot_bytes: bytes,
        cv_info: dict,
        created_at: Union[str, None] = None,
        updated_at: Union[str, None] = None,
    ):
        """
        Initialize a CrossValidationItem.

        Parameters
        ----------
        cv_results_serialized : dict
            The dict output of the :func:`sklearn.model_selection.cross_validate`
            function, in a form suitable for serialization.
        estimator_info : dict
            The estimator that was cross-validated.
        X_info : dict
            A summary of the data, input of the
            :func:`sklearn.model_selection.cross_validate` function.
        y_info : dict
            A summary of the target, input of the
            :func:`sklearn.model_selection.cross_validate` function.
        plot_bytes : bytes
            A plot of the cross-validation results, in the form of bytes.
        cv_info: dict
            A dict containing cross validation splitting strategy params.
        created_at : str
            The creation timestamp in ISO format.
        updated_at : str
            The last update timestamp in ISO format.
        """
        super().__init__(created_at, updated_at)

        self.cv_results_serialized = cv_results_serialized
        self.estimator_info = estimator_info
        self.X_info = X_info
        self.y_info = y_info
        self.plot_bytes = plot_bytes
        self.cv_info = cv_info

    def get_serializable_dict(self):
        """CrossValidationItem as a serializable dict."""
        # Get tabular results (the cv results in a dataframe-like structure)
        cv_results = copy.deepcopy(self.cv_results_serialized)
        cv_results.pop("indices", None)

        tabular_results = {
            "name": "Cross validation results",
            "columns": list(cv_results.keys()),
            "data": list(zip(*cv_results.values())),
        }

        # Get scalar results (summary statistics of the cv results)
        mean_cv_results = [
            {
                "name": _metric_title(k),
                "value": statistics.mean(v),
                "stddev": statistics.stdev(v),
            }
            for k, v in cv_results.items()
        ]

        scalar_results = mean_cv_results

        # Get estimator details (class name, parameters)
        _estimator_params = self.estimator_info["params"].items()

        # Figure out the default parameters of the estimator,
        # so that we can highlight the non-default ones in the UI

        # This is done by instantiating the class with no arguments and
        # computing the diff between the default and ours
        try:
            estimator_module = importlib.import_module(self.estimator_info["module"])
            EstimatorClass = getattr(estimator_module, self.estimator_info["name"])
            default_estimator_params = {
                k: repr(v) for k, v in EstimatorClass().get_params().items()
            }
        except Exception:
            default_estimator_params = {}

        estimator_params = {}
        for k, v in _estimator_params:
            if k in default_estimator_params and default_estimator_params[k] == v:
                estimator_params[k] = f"{v} (default)"
            else:
                estimator_params[k] = f"*{v}*"

        params_as_str = "\n".join(f"- {k}: {v}" for k, v in estimator_params.items())

        # If the estimator is from sklearn, make the class name a hyperlink
        # to the relevant docs
        name = self.estimator_info["name"]
        module = re.sub(r"\.\_.+", "", self.estimator_info["module"])
        if module.startswith("sklearn"):
            estimator_doc_target = f"https://scikit-learn.org/stable/modules/generated/{module}.{name}.html"
            estimator_doc_link = (
                f'<a href="{estimator_doc_target}" target="_blank">'
                f"<code>{name}</code></a>"
            )
        else:
            estimator_doc_link = f"`{name}`"

        estimator_params_as_str = f"{estimator_doc_link}\n{params_as_str}"

        # Get cross-validation details
        cv_params_as_str = ", ".join(f"{k}: *{v}*" for k, v in self.cv_info.items())

        d = super().get_serializable_dict()
        cv = {
            "scalar_results": scalar_results,
            "tabular_results": [tabular_results],
            "plots": [
                {
                    "name": "cross-validation results",
                    "value": json.loads(self.plot_bytes.decode("utf-8")),
                }
            ],
            "sections": [
                {
                    "title": "Model",
                    "icon": "icon-square-cursor",
                    "items": [
                        {
                            "name": "Estimator parameters",
                            "description": "Core model configuration used for training",
                            "value": estimator_params_as_str,
                        },
                        {
                            "name": "Cross-validation parameters",
                            "description": "Controls how data is split and validated",
                            "value": cv_params_as_str,
                        },
                    ],
                }
            ],
        }
        d.update(
            {
                "value": cv,
                "media_type": "application/vnd.skore.cross_validation+json",
            }
        )
        return d

    @classmethod
    def factory(cls, reporter):
        """
        Create a new CrossValidationItem instance from a CrossValidationReporter.

        Parameters
        ----------
        reporter : CrossValidationReporter

        Returns
        -------
        CrossValidationItem
            A new CrossValidationItem instance.
        """
        if not isinstance(reporter, CrossValidationReporter):
            raise ItemTypeError(
                f"Type '{reporter.__class__}' is not supported, "
                f"only '{CrossValidationReporter.__name__}' is."
            )

        cv_results = reporter._cv_results
        estimator = reporter.estimator
        X = reporter.X
        y = reporter.y
        plot = reporter.plot
        cv = reporter.cv

        cv_results_serialized = {}
        for k, v in cv_results.items():
            if k == "estimator":
                continue
            if k == "indices":
                cv_results_serialized["indices"] = {
                    "train": tuple(arr.tolist() for arr in v["train"]),
                    "test": tuple(arr.tolist() for arr in v["test"]),
                }
            if isinstance(v, numpy.ndarray):
                cv_results_serialized[k] = v.tolist()

        estimator_info = {
            "name": estimator.__class__.__name__,
            "module": estimator.__module__,
            "params": {k: repr(v) for k, v in estimator.get_params().items()},
        }

        y_array = y if isinstance(y, numpy.ndarray) else numpy.array(y)
        y_info = None if y is None else {"hash": _hash_numpy(y_array)}

        X_array = X if isinstance(X, numpy.ndarray) else numpy.array(X)
        X_info = {
            "nb_rows": X_array.shape[0],
            "nb_cols": X_array.shape[1],
            "hash": _hash_numpy(X_array),
        }

        plot_bytes = plotly.io.to_json(plot, engine="json").encode("utf-8")

        cv_info: dict[str, str] = {}
        if isinstance(cv, int):
            cv_info["n_splits"] = repr(cv)
        elif cv is None or hasattr(cv, "__iter__"):
            # cv is None or an iterable of splits
            cv_info["n_splits"] = "unknown"
        else:
            # cv is a sklearn CV splitter object
            for attr_name in ["n_splits", "shuffle", "random_state"]:
                with contextlib.suppress(AttributeError):
                    attr = getattr(cv, attr_name)
                    cv_info[attr_name] = repr(attr)

        return cls(
            cv_results_serialized=cv_results_serialized,
            estimator_info=estimator_info,
            X_info=X_info,
            y_info=y_info,
            plot_bytes=plot_bytes,
            cv_info=cv_info,
        )

    @cached_property
    def plot(self):
        """A plot of the cross-validation results."""
        return plotly.io.from_json(self.plot_bytes.decode("utf-8"))

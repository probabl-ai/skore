"""CrossValidationReporterItem.

This module defines the CrossValidationReporterItem class, which is used to persist
reporters of cross-validation.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import io
import json
import re
import statistics
from typing import TYPE_CHECKING, Literal, Optional, TypedDict

import joblib
import numpy
import plotly.graph_objects
import plotly.io

from skore.sklearn.cross_validation import CrossValidationReporter

from .item import Item, ItemTypeError

if TYPE_CHECKING:
    import sklearn.base

    class EstimatorParamInfo(TypedDict):
        """Information about an estimator parameter."""

        value: str
        default: bool

    class EstimatorInfo(TypedDict):
        """Information about an estimator."""

        name: str
        module: str
        params: dict[str, EstimatorParamInfo]


HUMANIZED_PLOT_NAMES = {
    "scores": "Scores",
    "timing": "Timings",
}


def _metric_title(metric):
    m = metric.replace("_", " ")
    title = f"Mean {m}"
    if title.endswith(" time"):
        title = title + " (seconds)"
    return title


def _metric_favorability(
    metric: str,
) -> Literal["greater_is_better", "lower_is_better", "unknown"]:
    greater_is_better_metrics = (
        "accuracy",
        "balanced_accuracy",
        "top_k_accuracy",
        "average_precision",
        "f1",
        "precision",
        "recall",
        "jaccard",
        "roc_auc",
        "r2",
    )
    any_match_greater_is_better = any(
        re.search(re.escape(pattern), metric) for pattern in greater_is_better_metrics
    )
    if (
        any_match_greater_is_better
        # other scikit-learn conventions
        or metric.endswith("_score")  # score: higher is better
        or metric.startswith("neg_")  # negative loss: negative of lower is better
    ):
        return "greater_is_better"

    lower_is_better_metrics = ("fit_time", "score_time")
    if (
        metric.endswith("_error")
        or metric.endswith("_loss")
        or metric.endswith("_deviance")
        or metric in lower_is_better_metrics
    ):
        return "lower_is_better"

    return "unknown"


def _params_to_str(estimator_info) -> str:
    params_list = []
    for k, v in estimator_info["params"].items():
        value = v["value"]
        if v["default"] is True:
            params_list.append(f"- {k}: {value} (default)")
        else:
            params_list.append(f"- {k}: *{value}*")

    return "\n".join(params_list)


def _estimator_info(estimator: sklearn.base.BaseEstimator) -> EstimatorInfo:
    estimator_params = (
        estimator.get_params() if hasattr(estimator, "get_params") else {}
    )

    name = estimator.__class__.__name__
    module = estimator.__module__

    # Figure out the default parameters of the estimator,
    # so that we can highlight the non-default ones in the UI

    # This is done by instantiating the class with no arguments and
    # computing the diff between the default and ours
    try:
        estimator_module = importlib.import_module(module)
        EstimatorClass = getattr(estimator_module, name)
        default_estimator_params = EstimatorClass().get_params()
    except Exception:
        default_estimator_params = {}

    final_estimator_params: dict[str, EstimatorParamInfo] = {}
    for k, v in estimator_params.items():
        param_is_default: bool = (
            k in default_estimator_params and default_estimator_params[k] == v
        )
        final_estimator_params[str(k)] = {
            "value": repr(v),
            "default": param_is_default,
        }

    return {
        "name": name,
        "module": module,
        "params": final_estimator_params,
    }


class CrossValidationReporterItem(Item):
    """Class to persist the reporter of cross-validation."""

    def __init__(
        self,
        reporter_bytes: bytes,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        note: Optional[str] = None,
    ):
        """
        Initialize a CrossValidationReporterItem.

        Parameters
        ----------
        reporter_bytes : bytes
            The raw bytes of the reporter pickled representation.
        created_at : str, optional
            The creation timestamp in ISO format.
        updated_at : str, optional
            The last update timestamp in ISO format.
        note : str, optional
            A note.
        """
        super().__init__(created_at, updated_at, note)

        self.reporter_bytes = reporter_bytes

    @classmethod
    def factory(
        cls,
        reporter: CrossValidationReporter,
        /,
        **kwargs,
    ) -> CrossValidationReporterItem:
        """
        Create a CrossValidationReporterItem instance from a CrossValidationReporter.

        Parameters
        ----------
        reporter : CrossValidationReporter

        Returns
        -------
        CrossValidationReporterItem
            A new CrossValidationReporterItem instance.
        """
        if not isinstance(reporter, CrossValidationReporter):
            raise ItemTypeError(f"Type '{reporter.__class__}' is not supported.")

        with io.BytesIO() as stream:
            joblib.dump(reporter, stream)

            return cls(stream.getvalue(), **kwargs)

    @property
    def reporter(self) -> CrossValidationReporter:
        """The CrossValidationReporter from the persistence."""
        with io.BytesIO(self.reporter_bytes) as stream:
            return joblib.load(stream)

    def as_serializable_dict(self):
        """Convert item to a JSON-serializable dict to used by frontend."""
        # Get tabular results (the cv results in a dataframe-like structure)
        cv_results = {
            key: value.tolist()
            for key, value in self.reporter._cv_results.items()
            if key not in ("estimator", "indices") and isinstance(value, numpy.ndarray)
        }

        metrics_names = list(cv_results)
        tabular_results = [
            {
                "name": "Cross validation results",
                "columns": metrics_names,
                "data": list(zip(*cv_results.values())),
                "favorability": [_metric_favorability(m) for m in metrics_names],
            }
        ]

        # Get scalar results (summary statistics of the cv results)
        scalar_results = [
            {
                "name": _metric_title(k),
                "value": statistics.mean(v),
                "stddev": statistics.stdev(v),
                "favorability": _metric_favorability(k),
            }
            for k, v in cv_results.items()
        ]

        # If the estimator is from sklearn, make the class name a hyperlink
        # to the relevant docs
        estimator_info = _estimator_info(self.reporter.estimator)
        name = estimator_info["name"]
        module = re.sub(r"\.\_.+", "", estimator_info["module"])
        if module.startswith("sklearn"):
            doc_url = f"https://scikit-learn.org/stable/modules/generated/{module}.{name}.html"
            doc_link = f'<a href="{doc_url}" target="_blank"><code>{name}</code></a>'
        else:
            doc_link = f"`{name}`"

        params_as_str = _params_to_str(estimator_info)
        estimator_params_as_str = f"{doc_link}\n{params_as_str}"

        # Serialize cross-validation settings
        cv = self.reporter.cv
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

        cv_params_as_str = ", ".join(f"{k}: *{v}*" for k, v in cv_info.items())

        # Construct the final representation
        value = {
            "scalar_results": scalar_results,
            "tabular_results": tabular_results,
            "plots": [
                {
                    "name": HUMANIZED_PLOT_NAMES[plot_name],
                    "value": json.loads(plotly.io.to_json(plot, engine="json")),
                }
                for plot_name, plot in dataclasses.asdict(self.reporter.plots).items()
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

        return super().as_serializable_dict() | {
            "media_type": "application/vnd.skore.cross_validation+json",
            "value": value,
        }

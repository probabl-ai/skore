"""Report, metric and artifact iteration utilities."""

from __future__ import annotations

import itertools
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from typing import Any, TypeAlias

import matplotlib.pyplot as plt
import mlflow.data
import narwhals as nw
import numpy as np
import pandas as pd
from mlflow.data.dataset import Dataset as MlFlowDatasetType
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone

from skore import CrossValidationReport, EstimatorReport
from skore._plugins import switch_plt_backend
from skore._utils.skrub import is_skrub_learner

ArrayLike: TypeAlias = pd.DataFrame | NDArray[np.generic]
InputExample: TypeAlias = ArrayLike | dict[str, Any]


@dataclass
class Artifact:
    """Artifact payload and target name."""

    name: str
    payload: Any


@dataclass
class Dataset:
    """Dataset metadata payload (wrapper over mlflow's dataset)."""

    dataset: MlFlowDatasetType
    context: str | None = None


@dataclass
class Metric:
    """Scalar metric payload."""

    name: str
    value: float


@dataclass
class Params:
    """Model parameter payload."""

    params: dict[str, Any]


@dataclass
class Tag:
    """Tag payload."""

    key: str
    value: str


@dataclass
class Model:
    """Model payload."""

    model: BaseEstimator
    input_example: InputExample


CLF_METRICS = {
    # metric -> kwargs
    "accuracy": {},
    "log_loss": {},
    "recall": {"average": "micro"},
    "precision": {"average": "micro"},
    "roc_auc": {"average": "micro", "multi_class": "ovr"},
}

REG_METRICS = {
    # metric -> kwargs
    "r2": {"multioutput": "uniform_average"},
    "rmse": {"multioutput": "uniform_average"},
}

# mappings per task type:
METRICS = {
    "binary-classification": CLF_METRICS,
    "multiclass-classification": CLF_METRICS,
    "regression": REG_METRICS,
    "multioutput-regression": REG_METRICS,
}

PLOTS = {
    "binary-classification": ["confusion_matrix", "roc", "precision_recall"],
    "multiclass-classification": ["confusion_matrix", "roc", "precision_recall"],
    "regression": ["prediction_error"],
    "multioutput-regression": [],
}


LogItem: TypeAlias = Params | Tag | Model | Artifact | Metric | Dataset
NestedLogItem: TypeAlias = LogItem | tuple[str, Iterable[LogItem]]


def iter_cv_metrics(
    report: CrossValidationReport,
) -> Generator[Artifact | Metric, Any, None]:
    """Yield metrics/artifacts for a cross-validation report."""
    ml_task = report.ml_task
    report_any = report

    for name, kwargs in METRICS[ml_task].items():
        method = getattr(report_any.metrics, name)
        yield Metric(name, method(**kwargs, aggregate="mean").iloc[0])
        yield Metric(f"{name}_std", method(**kwargs, aggregate="std").iloc[0])
        if not kwargs or ml_task == "regression":
            continue

    for name in PLOTS[ml_task]:
        method = getattr(report_any.metrics, name)
        display = method()
        yield Artifact(f"metrics_details/{name}", display.frame())
        with switch_plt_backend(), plt.ioff():
            figure = display.plot()
            if figure is None:
                # NOTE: backward compatibility for when `figure_` was stored as an
                # attribute in the display object instead of being returned by `plot`.
                figure = display.figure_
            try:
                yield Artifact(f"metrics.{name}", figure)
            finally:
                plt.close(figure)
        continue

    timings = report_any.metrics.timings()
    fit_time = timings.loc["Fit time (s)"].loc["mean"]
    fit_time_std = timings.loc["Fit time (s)"].loc["std"]
    predict_time = timings.loc["Predict time test (s)"].loc["mean"]
    predict_time_std = timings.loc["Predict time test (s)"].loc["std"]
    summary = report_any.metrics.summarize()

    yield Metric("fit_time", fit_time)
    yield Metric("fit_time_std", fit_time_std)
    yield Metric("predict_time", predict_time)
    yield Metric("predict_time_std", predict_time_std)
    # NOTE: auto format is wide for single CV reports; use long for per-split details.
    yield Artifact(
        "metrics_details/per_split",
        summary.frame(flat_index=False, aggregate=None),
    )
    yield Artifact("metrics", summary.frame())


def iter_estimator_metrics(
    report: EstimatorReport,
) -> Generator[Artifact | Metric, Any, None]:
    """Yield metrics/artifacts for an estimator report."""
    ml_task = report.ml_task
    report_any = report
    # NOTE: we could do the same things with data_source="train"

    for name, kwargs in METRICS[ml_task].items():
        method = getattr(report_any.metrics, name)
        yield Metric(name, method(**kwargs))

    for name in PLOTS[ml_task]:
        method = getattr(report_any.metrics, name)
        display = method()
        yield Artifact(f"metrics_details/{name}", display.frame())
        with switch_plt_backend(), plt.ioff():
            figure = display.plot()
            if figure is None:
                # NOTE: backward compatibility for when `figure_` was stored as an
                # attribute in the display object instead of being returned by `plot`.
                figure = display.figure_
            try:
                yield Artifact(f"metrics.{name}", figure)
            finally:
                plt.close(figure)
        continue

    timings = report_any.metrics.timings()
    yield Metric("fit_time", timings["fit_time"])
    yield Metric("predict_time", timings["predict_time_test"])
    yield Artifact("metrics", report_any.metrics.summarize().frame())


def iter_cv(report: CrossValidationReport) -> Generator[NestedLogItem, None, None]:
    """Yield loggable objects for a cross-validation report."""
    yield from iter_cv_metrics(report)

    estimator = clone(report.estimator_)
    if is_skrub_learner(estimator):
        estimator.fit(report.input_data)
        input_example: InputExample = _sample_environment(report.input_data)
    else:
        estimator.fit(report.X, report.y)
        input_example = _sample_input_example(report.X)
    yield Params(estimator.get_params())
    yield Model(estimator, input_example)

    yield Artifact("data.summarize", _data_analyze_html(report))

    yield _dataset_from_Xy(report.X, report.y)

    for split_id, estimator_report in enumerate(report.reports_):
        yield (
            f"split_{split_id}",
            itertools.chain(
                [Tag("split_id", str(split_id))], iter_estimator(estimator_report)
            ),
        )

    if report.splitter is not None:
        yield Params({"cv_splitter.class": report.splitter.__class__.__name__})

        try:
            n_splits = report.splitter.get_n_splits()
        except AttributeError:
            pass
        else:
            yield Params({"cv_splitter.n_splits": n_splits})


def iter_estimator(report: EstimatorReport) -> Generator[LogItem, None, None]:
    """Yield loggable objects for an estimator report."""
    yield from iter_estimator_metrics(report)

    estimator = report.estimator_
    if is_skrub_learner(estimator):
        test_data = report.test_data
        assert test_data is not None
        input_example: InputExample = _sample_environment(test_data)
    else:
        input_example = _sample_input_example(report.X_test)
    yield Params(estimator.get_params())
    yield Model(estimator, input_example)

    yield Artifact("data.summarize", _data_analyze_html(report))

    yield _dataset_from_Xy(report.X_train, report.y_train, context="training")
    yield _dataset_from_Xy(report.X_test, report.y_test, context="evaluation")


def _data_analyze_html(report: CrossValidationReport | EstimatorReport) -> Any:
    with switch_plt_backend(), plt.ioff():
        try:
            return report.data.summarize()._repr_html_()
        finally:
            plt.close("all")


def _pandas_from_polars(value: Any) -> Any:
    """Return ``value`` as pandas when it is a polars frame or series."""
    if nw.dependencies.is_polars_dataframe(value) or nw.dependencies.is_polars_series(
        value
    ):
        return nw.from_native(value, allow_series=True).to_pandas()
    return value


def _sample_environment(
    environment: dict[str, Any], *, max_samples: int = 5
) -> dict[str, Any]:
    """Row-sample an evaluation environment for an MLflow input example.

    Polars values are converted to pandas so signature inference sees a type
    MLflow already accepts. Scalars are left unchanged.
    """
    return {
        key: _sample_environment_value(value, max_samples=max_samples)
        for key, value in environment.items()
    }


def _sample_environment_value(value: Any, *, max_samples: int) -> Any:
    if isinstance(value, dict):
        return _sample_environment(value, max_samples=max_samples)
    value = _pandas_from_polars(value)
    if isinstance(value, pd.DataFrame | np.ndarray):
        return _sample_input_example(value, max_samples=max_samples)
    if isinstance(value, pd.Series):
        return value.head(max_samples)
    return value


def _sample_input_example(X: ArrayLike, *, max_samples: int = 5) -> ArrayLike:
    if isinstance(X, pd.DataFrame):
        sample = X.head(max_samples)
        category_columns = sample.select_dtypes(include="category").columns
        if len(category_columns) == 0:
            return sample
        # MLflow signature inference cannot always handle pandas categorical
        # dtypes. Cast to object to keep values while avoiding category dtype.
        sample = sample.copy()
        sample[category_columns] = sample[category_columns].astype(object)
        return sample
    else:
        return X[:max_samples]


def _dataset_from_polars(X: Any, y: Any, context: str | None) -> Dataset:
    """Log a polars feature frame with ``mlflow.data.from_polars``.

    ``from_polars`` records a single target column. Several targets fall back to
    the numpy dataset path.
    """
    import polars as pl

    if isinstance(y, dict) or (
        isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] != 1
    ):
        if isinstance(y, dict):
            targets = y
        else:
            targets = {f"target_{idx}": y[:, idx] for idx in range(y.shape[1])}
        return _dataset_from_Xy(X.to_numpy(), targets, context=context)

    if isinstance(y, pl.Series):
        name = y.name if y.name else "target"
        target_frame = y.alias(name).to_frame()
    elif isinstance(y, pl.DataFrame):
        if y.width != 1:
            return _dataset_from_Xy(
                X.to_numpy(),
                {column: y.get_column(column).to_numpy() for column in y.columns},
                context=context,
            )
        name = str(y.columns[0])
        target_frame = y
    elif isinstance(y, pd.Series):
        name = str(y.name) if y.name is not None else "target"
        target_frame = pl.Series(name, y.to_numpy()).to_frame()
    elif isinstance(y, pd.DataFrame):
        if len(y.columns) != 1:
            return _dataset_from_Xy(
                X.to_numpy(),
                {column: y[column].to_numpy() for column in y.columns},
                context=context,
            )
        name = str(y.columns[0])
        target_frame = pl.from_pandas(y)
    elif isinstance(y, np.ndarray):
        name = "target"
        values = y.ravel() if y.ndim == 2 else y
        target_frame = pl.Series(name, values).to_frame()
    else:
        raise TypeError(f"Unsupported target type for a polars dataset: {type(y)}")

    frame = pl.concat([X, target_frame], how="horizontal")
    return Dataset(
        dataset=mlflow.data.from_polars(frame, targets=name),  # type: ignore[attr-defined]
        context=context,
    )


def _dataset_from_Xy(
    X: pd.DataFrame | NDArray[np.generic],
    y: pd.DataFrame | pd.Series | NDArray[np.generic] | dict[str, NDArray[np.generic]],
    context: str | None = None,
) -> Dataset:
    if nw.dependencies.is_polars_dataframe(X):
        return _dataset_from_polars(X, y, context)

    if isinstance(X, np.ndarray):
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        elif isinstance(y, pd.DataFrame):
            y = {column: y[column].to_numpy() for column in y.columns}

        return Dataset(
            dataset=mlflow.data.from_numpy(X, targets=y),  # type: ignore[attr-defined]
            context=context,
        )

    if isinstance(y, np.ndarray):
        if y.ndim == 1:
            y = pd.Series(y, index=X.index, name="target")
        else:
            y = pd.DataFrame(
                y, index=X.index, columns=[f"target_{idx}" for idx in range(y.shape[1])]
            )

    assert isinstance(y, (pd.DataFrame, pd.Series))

    if isinstance(y, pd.Series):
        name = str(y.name) if y.name is not None else "target"
        targets = name
        y = pd.DataFrame({name: y})
    elif len(y.columns) == 1:
        targets = y.columns[0]
    else:
        # mlflow.data.from_pandas doesn't support multiple targets
        # use mlflow.data.from_numpy instead
        return _dataset_from_Xy(
            X.to_numpy(), {c: y[c].to_numpy() for c in y.columns}, context=context
        )

    Xy = pd.concat([X, y], axis=1)
    mlflow_dataset = mlflow.data.from_pandas(Xy, targets=targets)  # type: ignore[attr-defined]
    return Dataset(dataset=mlflow_dataset, context=context)

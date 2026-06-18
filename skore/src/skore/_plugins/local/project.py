import contextlib
import json
import os
import pickle
import shutil
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ... import CrossValidationReport, EstimatorReport
from ..._utils._cache_key import deep_key_sanitize


def init_workspace(workspace_dir: str | Path, project_name: str = "default") -> Path:
    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / ".SKORE_WORKSPACE").touch()
    (workspace_dir / "projects").mkdir(exist_ok=True)
    (workspace_dir / "datasets").mkdir(exist_ok=True)
    return workspace_dir


def find_workspace() -> Path:
    start = Path(".").resolve()
    for candidate in [start, *start.parents[::-1]]:
        workspace = candidate / "skore_data"
        if workspace.is_dir() and (workspace / ".SKORE_WORKSPACE").exists():
            return workspace
    if env_workspace := os.environ.get("SKORE_WORKSPACE"):
        return init_workspace(Path(env_workspace))
    return init_workspace(Path.home() / "skore_data")


def _init_project_dir(workspace: Path, project_name: str) -> Path:
    project_dir = workspace / "projects" / project_name
    if (
        not project_dir.expanduser()
        .resolve()
        .is_relative_to(workspace.expanduser().resolve() / "projects")
    ):
        raise ValueError(project_name)
    if project_dir.is_dir():
        return project_dir
    project_dir.mkdir(parents=True)
    (project_dir / "reports").mkdir()
    return project_dir


def _dump_report(
    report: EstimatorReport,
    *,
    workspace: Path | None = None,
    project_name: str = "default",
    name: str | None = None,
) -> Path:
    workspace = find_workspace() if workspace is None else Path(workspace)
    project_dir = _init_project_dir(workspace, project_name)
    reports_dir = project_dir / "reports"
    date_str = str(report._metadata["creation-date"]).replace(":", "-")
    name_str = "" if name is None else f"__{name}"
    output_dir = (
        reports_dir / f"{date_str}__{report._metadata['id']:x}__"
        f"{report._metadata['report_type']}{name_str}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    symlink = reports_dir / f"latest{name_str}"
    with contextlib.suppress(FileNotFoundError):
        symlink.unlink()
    with contextlib.suppress(OSError):
        symlink.symlink_to(output_dir)

    if isinstance(report, EstimatorReport):
        _dump_estimator_report(report, workspace, output_dir, name=name)
    else:
        _write_report_contents(report, output_dir, workspace, name)
        _write_cv_split_reports(report, output_dir, workspace)
    return output_dir


def _write_cv_split_reports(
    report: CrossValidationReport, output_dir: Path, workspace: Path
) -> None:
    split_reports_dir = output_dir / "reports"
    split_reports_dir.mkdir()
    n_digits = int(np.ceil(np.log10(len(report.reports_))))
    for split, (sub_report, (train_idx, test_idx)) in enumerate(
        zip(report.reports_, report._split_indices, strict=True)
    ):
        split_dir = split_reports_dir / f"split_{{:0>{n_digits}}}".format(split)
        split_dir.mkdir()
        _dump_estimator_report(sub_report, workspace, split_dir)
        np.savetxt(split_dir / "train_indices.txt", train_idx, fmt="%d")
        np.savetxt(split_dir / "test_indices.txt", test_idx, fmt="%d")


def _write_metadata(
    report: EstimatorReport | CrossValidationReport,
    output_dir: Path,
    name: str | None = None,
) -> None:
    metadata = report._metadata | {
        "ml_task": report._ml_task,
        "learner": repr(report.estimator_),
    }
    if name is not None:
        metadata["name"] = name
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), "UTF-8")


def _write_report_state(
    report: EstimatorReport | CrossValidationReport, output_dir: Path
) -> None:
    state = report.to_dict()
    (output_dir / "state.json").write_text(
        json.dumps(
            {
                k: v
                for k, v in state.items()
                if k
                not in (
                    "estimator",
                    "learner",
                    "data",
                    "predictions",
                    "metric_registry",
                    "optional",
                    "split_indices",
                    "estimator_reports",
                )
            }
            | {"export-format-version": 1}
        ),
        "UTF-8",
    )


def _write_metrics(
    report: EstimatorReport | CrossValidationReport, output_dir: Path
) -> None:
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    report.metrics.summarize().data.to_csv(metrics_dir / "summarize.csv", index=False)
    if hasattr(report, "_metric_registry"):
        with open(metrics_dir / "registry.pickle", "wb") as f:
            pickle.dump(report._metric_registry, f)


def _write_datasets(
    report: EstimatorReport | CrossValidationReport, output_dir: Path, workspace: Path
) -> None:
    dataset_refs_dir = output_dir / "data"
    dataset_refs_dir.mkdir(exist_ok=True)

    for subset_name in ["_data", "_train_data", "_test_data"]:
        if (subset := getattr(report, subset_name, None)) is not None:
            subset_refs = {}
            if subset is not None:
                for key, val in subset.items():
                    subset_refs[key] = _get_data_ref(val, workspace)
                refs_file = dataset_refs_dir / f"{subset_name.removeprefix('_')}.json"
                refs_file.write_text(json.dumps(subset_refs), "UTF-8")


def _write_estimators(
    report: EstimatorReport | CrossValidationReport, output_dir: Path
) -> None:
    with open(output_dir / "estimator.pickle", "wb") as f:
        pickle.dump(report.estimator, f)
    with open(output_dir / "estimator_.pickle", "wb") as f:
        pickle.dump(report.estimator_, f)
    with open(output_dir / "learner_.pickle", "wb") as f:
        pickle.dump(report.learner_, f)


def _make_user_dir(output_dir: Path) -> None:
    user_dir = output_dir / "user"
    user_dir.mkdir(exist_ok=True)
    (user_dir / "README").write_text(
        "This directory is not used by skore, use it to store arbitrary "
        "additional data or notes attached to this report.\n"
    )


def _write_checks(
    report: EstimatorReport | CrossValidationReport, output_dir: Path
) -> None:
    checks_dir = output_dir / "checks"
    checks_dir.mkdir(exist_ok=True)
    report.checks.summarize(fast_mode=True).frame().to_csv(
        checks_dir / "summarize.csv", index=False
    )
    with open(checks_dir / "_check_results_cache.json", "w", encoding="UTF-8") as f:
        json.dump(getattr(report, "_check_results_cache", {}), f)


def _write_report_contents(
    report: EstimatorReport | CrossValidationReport,
    output_dir: Path,
    workspace: Path,
    name: str | None,
) -> None:
    _write_metadata(report, output_dir, name=name)
    _write_report_state(report, output_dir)
    _write_estimators(report, output_dir)
    _make_user_dir(output_dir)
    _write_metrics(report, output_dir)
    _write_datasets(report, output_dir, workspace)
    _write_checks(report, output_dir)


def _write_permutation_importances(report: EstimatorReport, output_dir: Path) -> None:
    importances_dir = output_dir / "inspection" / "permutation_importance"
    importances_dir.mkdir(exist_ok=True, parents=True)
    for cache_item in report._cache.items():
        match cache_item:
            case (
                data_source,
                "permutation_importance",
                ("mapping", kwarg_items),
            ), display:
                kwargs = dict(kwarg_items)
                display_dir = (
                    importances_dir / "permutation_importance__"
                    f"{data_source}__{kwargs['at_step']}__{kwargs['metric']}"
                )
                display_dir.mkdir(exist_ok=True)
                display.importances.to_csv(display_dir / "importances.csv", index=False)
                (display_dir / "kwargs.json").write_text(json.dumps(kwargs), "UTF-8")
                (display_dir / "cache_key.json").write_text(
                    json.dumps((data_source, "permutation_importance", kwargs))
                )
            case _:
                pass


def _dump_estimator_report(
    report: EstimatorReport,
    workspace: Path,
    output_dir: Path,
    name: str | None = None,
) -> Path:
    _write_report_contents(report, output_dir, workspace, name)
    _write_permutation_importances(report, output_dir)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    for (subset_name, meth_name), val in report.to_dict()["predictions"].items():
        with open(predictions_dir / f"{subset_name}__{meth_name}.joblib", "wb") as f:
            joblib.dump(val, f)

    return output_dir


def load_report_metadata(report_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = json.loads(
        (report_dir / "metadata.json").read_text("UTF-8")
    )
    metadata["date"] = metadata["creation-date"]
    metadata["key"] = metadata["name"]
    if metadata["report_type"] == "cross-validation":
        subset_name = "data"
    else:
        subset_name = "test_data"
    metadata["dataset"] = json.loads(
        (report_dir / "data" / f"{subset_name}.json").read_text("UTF-8")
    )["_skrub_y"]["hash"]
    metrics = pd.read_csv(report_dir / "metrics" / "summarize.csv")
    metadata |= metrics.groupby("metric_name")["score"].mean().to_dict()
    return metadata


def load_report(report_dir: Path) -> EstimatorReport | CrossValidationReport:
    metadata = json.loads((report_dir / "metadata.json").read_text("UTF-8"))
    state = json.loads((report_dir / "state.json").read_text("UTF-8"))
    with open(report_dir / "estimator.pickle", "rb") as f:
        state["estimator"] = pickle.load(f)
    with open(report_dir / "learner_.pickle", "rb") as f:
        state["learner"] = pickle.load(f)
    data_dict = (
        state
        if metadata["report_type"] == "cross-validation"
        else state.setdefault("data", {})
    )
    for data_info_file in (report_dir / "data").glob("*.json"):
        data_info = json.loads(data_info_file.read_text("UTF-8"))
        loaded_data = {}
        for k, v in data_info.items():
            with open(v["file_path"], "rb") as f:
                loaded_data[k] = joblib.load(f)
        data_dict[data_info_file.stem] = loaded_data

    check_results_cache = json.loads(
        (report_dir / "checks" / "_check_results_cache.json").read_text("UTF-8")
    )
    if metadata["report_type"] == "cross-validation":
        sub_reports = []
        split_indices = []
        for p in sorted((report_dir / "reports").glob("split_*")):
            sub_reports.append(load_report(p).to_dict())
            split_indices.append(
                (
                    np.loadtxt(p / "train_indices.txt", dtype=int),
                    np.loadtxt(p / "test_indices.txt", dtype=int),
                )
            )
        state["estimator_reports"] = sub_reports
        state["split_indices"] = split_indices
        loaded_cv_report = CrossValidationReport.from_dict(state)
        loaded_cv_report._check_results_cache = check_results_cache
        return loaded_cv_report
    state["predictions"] = {}
    for pred_file in (report_dir / "predictions").glob("*.joblib"):
        with open(pred_file, "rb") as f:
            state["predictions"][tuple(pred_file.stem.split("__"))] = joblib.load(f)
    with open(report_dir / "metrics" / "registry.pickle", "rb") as f:
        state["metric_registry"] = pickle.load(f)
    state["optional"] = {"cache": {}}
    for importances_dir in (report_dir / "inspection" / "permutation_importance").glob(
        "permutation_importance__*"
    ):
        key = deep_key_sanitize(
            json.loads((importances_dir / "cache_key.json").read_text("UTF-8"))
        )
        value = pd.read_csv(importances_dir / "importances.csv")
        state["optional"]["cache"][key] = value
    loaded_report = EstimatorReport.from_dict(state)
    loaded_report._check_results_cache = check_results_cache
    return loaded_report


def _get_data_ref(value: Any, workspace: Path) -> dict[str, str]:
    h = joblib.hash(value)
    file_name = f"{h}.joblib"
    target = workspace / "datasets" / file_name
    if not target.is_file():
        with open(target, "wb") as f:
            joblib.dump(value, f)
    return {"hash": h, "file_name": file_name, "file_path": str(target)}


class Project:
    def __init__(self, name: str, workspace: str | Path | None = None):
        self.name = name
        self.workspace = (
            find_workspace() if workspace is None else init_workspace(workspace)
        )
        self.path = _init_project_dir(self.workspace, self.name)

    def put(self, key: str, report: EstimatorReport) -> Path:
        return _dump_report(
            report, workspace=self.workspace, project_name=self.name, name=key
        )

    def get(self, report_id: str) -> EstimatorReport | CrossValidationReport:
        report_path = next(
            iter((self.path / "reports").glob(f"*{int(report_id):x}*")), None
        )
        if report_path is None:
            raise KeyError(report_id)
        return load_report(report_path)

    def summarize(self) -> list[dict[str, Any]]:
        reports_path = self.path / "reports"
        result = []
        for p in sorted(reports_path.iterdir()):
            try:
                result.append(load_report_metadata(p))
            except Exception as e:
                warnings.warn(f"Failed to load report at {p}: {e!r}", stacklevel=2)
                raise
        return result

    @staticmethod
    def delete(*, name: str, workspace: str | Path) -> None:
        workspace = Path(workspace)
        if not (workspace / ".SKORE_WORKSPACE").exists():
            raise ValueError(f"Not a skore workspace: {workspace}")
        path = Project(name, workspace).path
        shutil.rmtree(path)

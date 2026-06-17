import contextlib
import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ... import EstimatorReport


def init_data_dir(parent_dir: str | Path = ".", project_name: str = "default") -> Path:
    data_dir = Path(parent_dir) / "skore_data"
    if data_dir.is_dir():
        return data_dir
    data_dir.mkdir(parents=True)
    (data_dir / ".SKORE_DATA_DIRECTORY").touch()
    (data_dir / "projects").mkdir()
    (data_dir / "datasets").mkdir()
    return data_dir


def find_data_dir() -> Path:
    start = Path(".").resolve()
    for candidate in [start, *start.parents[::-1]]:
        data_dir = candidate / "skore_data"
        if data_dir.is_dir() and (data_dir / ".SKORE_DATA_DIRECTORY").exists():
            return data_dir
    return init_data_dir(Path.home())


def init_project_dir(data_dir: Path, project_name: str) -> Path:
    project_dir = data_dir / "projects" / project_name
    if project_dir.is_dir():
        return project_dir
    project_dir.mkdir(parents=True)
    (project_dir / "reports").mkdir()
    return project_dir


def export(
    report: EstimatorReport,
    *,
    data_dir: Path | None = None,
    project_name: str = "default",
    name: str | None = None,
) -> Path:
    data_dir = find_data_dir() if data_dir is None else Path(data_dir)
    project_dir = init_project_dir(data_dir, project_name)
    reports_dir = project_dir / "reports"
    date_str = (
        datetime.fromisoformat(str(report._metadata["creation-date"]))
        .replace(tzinfo=None)
        .isoformat()
        .replace(":", "-")
    )
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
    (output_dir / "metadata.json").write_text(
        json.dumps(
            report._metadata
            | {
                "ml_task": report._ml_task,
                "learner": repr(report.estimator_),
                "name": name,
            },
            indent=2,
        ),
        "UTF-8",
    )
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
                )
            }
            | {"export-format-version": 1}
        ),
        "UTF-8",
    )
    with open(output_dir / "estimator.pickle", "wb") as f:
        pickle.dump(report.estimator, f)
    with open(output_dir / "estimator_.pickle", "wb") as f:
        pickle.dump(report.estimator_, f)
    with open(output_dir / "learner_.pickle", "wb") as f:
        pickle.dump(report.learner_, f)

    user_dir = output_dir / "user"
    user_dir.mkdir(exist_ok=True)
    (user_dir / "README").write_text(
        "This directory is not used by skore, use it to store arbitrary "
        "additional data or notes attached to this report.\n"
    )

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    report.metrics.summarize().frame(flat_index=True).to_csv(
        metrics_dir / "summarize.csv"
    )
    with open(metrics_dir / "registry.pickle", "wb") as f:
        pickle.dump(report._metric_registry, f)

    checks_dir = output_dir / "checks"
    checks_dir.mkdir(exist_ok=True)
    report.checks.summarize().frame().to_csv(checks_dir / "summarize.csv", index=False)

    dataset_refs_dir = output_dir / "data"
    dataset_refs_dir.mkdir(exist_ok=True)

    for subset_name, subset in state["data"].items():
        subset_refs = {}
        if subset is not None:
            for key, val in subset.items():
                subset_refs[key] = get_data_ref(val, data_dir)
            refs_file = dataset_refs_dir / f"{subset_name.removeprefix('_')}.json"
            refs_file.write_text(json.dumps(subset_refs), "UTF-8")

    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    for (subset_name, meth_name), val in report.to_dict()["predictions"].items():
        with open(predictions_dir / f"{subset_name}__{meth_name}.joblib", "wb") as f:
            joblib.dump(val, f)

    inspection_dir = output_dir / "inspection"
    inspection_dir.mkdir(exist_ok=True)
    permutation_displays = [
        (k, v) for k, v in report._cache.items() if k[1] == "permutation_importance"
    ]
    for k, display in permutation_displays:
        display_dir = inspection_dir / f"{k[0]}__{k[2][1][0][1]}"
        # TODO handle multiple permutations for the same thing: keep the one
        # with most repeats, add suffix to dir, ...
        #
        # Store creation params in cache and in display object
        display_dir.mkdir(exist_ok=True)
        display.importances.to_csv(display_dir / "importances.csv", index=False)
    return output_dir


def load_metadata(report_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = json.loads(
        (report_dir / "metadata.json").read_text("UTF-8")
    )
    metadata["date"] = metadata["creation-date"]
    metadata["key"] = metadata["name"]
    metadata["dataset"] = json.loads(
        (report_dir / "data" / "test_data.json").read_text("UTF-8")
    )["_skrub_y"]["hash"]
    metrics = pd.read_csv(report_dir / "metrics" / "summarize.csv")
    metadata |= metrics.set_index("Metric").iloc[:, 0].to_dict()
    return metadata


def load(report_dir: Path) -> EstimatorReport:
    state = json.loads((report_dir / "state.json").read_text("UTF-8"))
    with open(report_dir / "estimator.pickle", "rb") as f:
        state["estimator"] = pickle.load(f)
    with open(report_dir / "learner_.pickle", "rb") as f:
        state["learner"] = pickle.load(f)
    state["predictions"] = {}
    for pred_file in (report_dir / "predictions").glob("*.joblib"):
        with open(pred_file, "rb") as f:
            state["predictions"][tuple(pred_file.stem.split("__"))] = joblib.load(f)
    with open(report_dir / "metrics" / "registry.pickle", "rb") as f:
        state["metric_registry"] = pickle.load(f)
    state["data"] = {}
    for data_info_file in (report_dir / "data").glob("*.json"):
        data_info = json.loads(data_info_file.read_text("UTF-8"))
        loaded_data = {}
        for k, v in data_info.items():
            with open(v["file_path"], "rb") as f:
                loaded_data[k] = -joblib.load(f)
        state["data"][data_info_file.stem] = loaded_data
    state["optional"] = {"cache": {}}
    return EstimatorReport.from_dict(state)


def get_data_ref(value: Any, data_dir: Path) -> dict[str, str]:
    h = joblib.hash(value)
    file_name = f"{h}.joblib"
    target = data_dir / "datasets" / file_name
    if not target.is_file():
        with open(target, "wb") as f:
            joblib.dump(value, f)
    return {"hash": h, "file_name": file_name, "file_path": str(target)}


class Project:
    def __init__(self, name: str, workspace: str | Path | None = None):
        self.name = name
        self.workspace = workspace

    def data_dir(self) -> Path:
        return (
            find_data_dir()
            if self.workspace is None
            else init_data_dir(Path(self.workspace))
        )

    def path(self) -> Path:
        return init_project_dir(self.data_dir(), self.name)

    def put(self, key: str, report: EstimatorReport) -> Path:
        return export(
            report, data_dir=self.data_dir(), project_name=self.name, name=key
        )

    def get(self, report_id: int) -> EstimatorReport:
        report_path = next(
            iter((self.path() / "reports").glob(f"*{report_id:x}*")), None
        )
        if report_path is None:
            raise KeyError(report_id)
        return load(report_path)

    def summarize(self) -> list[dict[str, Any]]:
        reports_path = self.path() / "reports"
        return [load_metadata(p) for p in reports_path.iterdir()]

    def delete(self) -> None:
        shutil.rmtree(self.path())

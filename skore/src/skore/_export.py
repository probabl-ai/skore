import contextlib
import json
import pickle
from datetime import datetime
from pathlib import Path

import joblib

from . import EstimatorReport


def init_data_dir(parent_dir="."):
    data_dir = Path(parent_dir) / "skore_data"
    if data_dir.is_dir():
        return data_dir
    data_dir.mkdir()
    (data_dir / ".SKORE_DATA_DIRECTORY").touch()
    (data_dir / "datasets").mkdir()
    (data_dir / "reports").mkdir()
    return data_dir


def find_data_dir():
    start = Path(".").resolve()
    for candidate in [start, *start.parents[::-1]]:
        data_dir = candidate / "skore_data"
        if data_dir.is_dir() and (data_dir / ".SKORE_DATA_DIRECTORY").exists():
            return data_dir
    return init_data_dir(Path.home())


def export(report, *, root_data_dir=None, name=None):
    root_data_dir = find_data_dir() if root_data_dir is None else Path(root_data_dir)
    reports_dir = root_data_dir / "reports"
    date_str = (
        datetime.fromisoformat(report._metadata["creation-date"])
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
        json.dumps(report._metadata, indent=2), "UTF-8"
    )
    state = report.get_state()
    (output_dir / "state.json").write_text(
        json.dumps(
            {
                k: v
                for k, v in state.items()
                if k
                not in (
                    "raw_estimator",
                    "estimator",
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
        pickle.dump(report.estimator_, f)

    # for from_state, but we can probably get rid of _raw_estimator
    with open(output_dir / "_estimator.pickle", "wb") as f:
        pickle.dump(report._estimator, f)
    with open(output_dir / "_raw_estimator.pickle", "wb") as f:
        pickle.dump(report._raw_estimator, f)

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

    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    for subset_name, subset in state["data"].items():
        subset_refs = {}
        if subset is not None:
            for key, val in subset.items():
                subset_refs[key] = get_data_ref(val, root_data_dir)
            refs_file = data_dir / f"{subset_name.removeprefix('_')}.json"
            refs_file.write_text(json.dumps(subset_refs), "UTF-8")

    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    for (subset_name, meth_name), val in report.get_state()["predictions"].items():
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


def load(report_dir):
    report_dir = Path(report_dir)
    state = json.loads((report_dir / "state.json").read_text("UTF-8"))
    with open(report_dir / "_estimator.pickle", "rb") as f:
        state["estimator"] = pickle.load(f)
    with open(report_dir / "_raw_estimator.pickle", "rb") as f:
        state["raw_estimator"] = pickle.load(f)
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
    return EstimatorReport.from_state(state)


def get_data_ref(value, root_data_dir):
    h = joblib.hash(value)
    file_name = f"{h}.joblib"
    target = root_data_dir / "datasets" / file_name
    if not target.is_file():
        with open(target, "wb") as f:
            joblib.dump(value, f)
    return {"hash": h, "file_name": file_name, "file_path": str(target)}

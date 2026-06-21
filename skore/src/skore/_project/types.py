from typing import Literal, TypedDict

ProjectMode = Literal["hub", "local", "mlflow"]
PluginGroup = Literal["skore.plugins.project", "skore.plugins.login"]
SyncDirection = Literal["put", "get", "both"]
ConflictPolicy = Literal[
    "latest_wins",
    "source_wins",
    "destination_wins",
    "skip",
    "keep_both",
    "error",
]


class ReportMetadata(TypedDict):
    """Metadata and metrics for a single persisted report."""

    id: str
    key: str
    date: str
    learner: str
    ml_task: str
    report_type: str
    dataset: str
    rmse: float | None
    log_loss: float | None
    roc_auc: float | None
    fit_time: float | None
    predict_time: float | None
    rmse_mean: float | None
    log_loss_mean: float | None
    roc_auc_mean: float | None
    fit_time_mean: float | None
    predict_time_mean: float | None
    rmse_std: float | None
    log_loss_std: float | None
    roc_auc_std: float | None
    fit_time_std: float | None
    predict_time_std: float | None

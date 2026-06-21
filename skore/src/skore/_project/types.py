from typing import Literal

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

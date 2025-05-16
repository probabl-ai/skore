"""Utility functions for Skore and Scikit-learn integration."""

_SCORE_OR_LOSS_INFO: dict[str, dict[str, str]] = {
    "fit_time": {"name": "Fit time (s)", "icon": "(↘︎)"},
    "predict_time": {"name": "Predict time (s)", "icon": "(↘︎)"},
    "accuracy": {"name": "Accuracy", "icon": "(↗︎)"},
    "precision": {"name": "Precision", "icon": "(↗︎)"},
    "recall": {"name": "Recall", "icon": "(↗︎)"},
    "brier_score": {"name": "Brier score", "icon": "(↘︎)"},
    "roc_auc": {"name": "ROC AUC", "icon": "(↗︎)"},
    "log_loss": {"name": "Log loss", "icon": "(↘︎)"},
    "r2": {"name": "R²", "icon": "(↗︎)"},
    "rmse": {"name": "RMSE", "icon": "(↘︎)"},
    "custom_metric": {"name": "Custom metric", "icon": ""},
    "report_metrics": {"name": "Report metrics", "icon": ""},
}

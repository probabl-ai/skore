from skore.externals._pandas_accessors import DirNamesMixin
from skore.sklearn._base import _BaseAccessor

###############################################################################
# Metrics accessor
###############################################################################


class _MetricsAccessor(_BaseAccessor, DirNamesMixin):
    """Accessor for metrics-related operations.

    You can access this accessor using the `metrics` attribute.
    """

    _SCORE_OR_LOSS_ICONS = {
        "accuracy": "(↗︎)",
        "precision": "(↗︎)",
        "recall": "(↗︎)",
        "brier_score": "(↘︎)",
        "roc_auc": "(↗︎)",
        "log_loss": "(↘︎)",
        "r2": "(↗︎)",
        "rmse": "(↘︎)",
        "report_metrics": "",
        "custom_metric": "",
    }

    def __init__(self, parent):
        super().__init__(parent, icon=":straight_ruler:")


########################################################################################
# Sub-accessors
# Plotting
########################################################################################


class _PlotMetricsAccessor(_BaseAccessor):
    """Plotting methods for the metrics accessor."""

    def __init__(self, parent):
        super().__init__(parent._parent, icon=":art:")
        self._metrics_parent = parent

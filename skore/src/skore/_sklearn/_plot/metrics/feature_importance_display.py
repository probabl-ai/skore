from skore._sklearn._plot.base import Display
from skore._sklearn._plot.style import StyleDisplayMixin
from skore._sklearn._plot.utils import HelpDisplayMixin, PlotBackendMixin


class FeatureImportanceDisplay(
    HelpDisplayMixin, StyleDisplayMixin, PlotBackendMixin, Display
):
    def __init__(self, _parent):
        self._parent = _parent

    def frame(self):
        """Return the coefficients as a dataframe."""
        return self._parent._dispatch_coefficient_frame()

    # @singledispatchmethod
    @StyleDisplayMixin.style_plot
    def _plot_matplotlib(self, **kwargs):
        return self._parent._dispatch_coefficient_plot(**kwargs)

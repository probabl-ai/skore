from sklearn.metrics import PredictionErrorDisplay

from skore.sklearn._plot.utils import HelpDisplayMixin


class PredictionErrorDisplay(HelpDisplayMixin, PredictionErrorDisplay):
    pass

from sklearn.metrics import PredictionErrorDisplay

from skore.sklearn._plot.utils import HelpDisplayMixin


# FIXME: we should write our own class here to not expose the class methods from
# scikit-learn and only relied on the estimator predictions.
class PredictionErrorDisplay(HelpDisplayMixin, PredictionErrorDisplay):
    pass

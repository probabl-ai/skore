from sklearn.linear_model import LinearRegression
from skrub import tabular_pipeline

from skore import evaluate
from skore._sklearn._checks._utils import get_preprocessed_X


def test_get_preprocessed_X_tabular_pipeline_with_numpy(regression_data):
    X, y = regression_data
    report = evaluate(tabular_pipeline(LinearRegression()), X, y, splitter=0.2)
    preprocessed = get_preprocessed_X(report, data_source="train")
    assert preprocessed is not None
    assert preprocessed.shape[0] == report.X_train.shape[0]

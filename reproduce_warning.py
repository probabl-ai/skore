from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from skore import ComparisonReport, CrossValidationReport
import matplotlib.pyplot as plt
import warnings

# Ensure warnings are shown
warnings.simplefilter("always")

X, y = load_iris(return_X_y=True)
estimator_1 = LogisticRegression(max_iter=10000, random_state=42)
estimator_2 = Pipeline(
    [
        ("poly", PolynomialFeatures()),
        ("predictor", LogisticRegression(max_iter=10000, random_state=0)),
    ]
)
report = ComparisonReport(
    [CrossValidationReport(estimator_1, X, y), CrossValidationReport(estimator_2, X, y)]
)
display = report.inspection.coefficients()
print("Plotting coefficients...")
display.plot()
plt.close('all')
print("Done.")

import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from skore import evaluate
from skore._externals._sklearn_compat import convert_container


@pytest.mark.parametrize(
    "x_container,y_container",
    [
        ("array", "array"),
        ("pandas", "series"),
        ("polars", "polars_series"),
    ],
)
@pytest.mark.filterwarnings("ignore:X does not have valid feature names:UserWarning")
def test_detects_underrepresented_classes(x_container, y_container):
    """Check that the underrepresented classes issue is detected."""
    weights = [0.9, 0.05, 0.05]
    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=3,
        n_classes=len(weights),
        random_state=0,
    )
    report = evaluate(LogisticRegression(), X, y, splitter=0.2)
    result = report.checks.summarize()
    assert "SKD005" not in set(result.frame(section="issue")["code"])

    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=3,
        n_classes=len(weights),
        weights=weights,
        random_state=0,
    )
    feature_columns = [str(i) for i in range(X.shape[1])]
    X = convert_container(X, x_container, column_names=feature_columns)
    y = convert_container(y, y_container)
    report = evaluate(LogisticRegression(), X, y, splitter=0.2)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD005" in issues.index
    assert "Accuracy should not be used alone" in issues.loc["SKD005", "explanation"]


@pytest.mark.parametrize(
    "x_container,y_container",
    [
        ("array", "array"),
        ("pandas", "series"),
        ("polars", "polars_series"),
    ],
)
@pytest.mark.filterwarnings("ignore:X does not have valid feature names:UserWarning")
def test_detects_underrepresented_classes_cv(x_container, y_container):
    """Check that underrepresented classes are detected on a cross-validation report."""
    weights = [0.9, 0.05, 0.05]
    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=3,
        n_classes=len(weights),
        n_clusters_per_class=1,
        weights=weights,
        random_state=0,
    )
    feature_columns = [str(i) for i in range(X.shape[1])]
    X = convert_container(X, x_container, column_names=feature_columns)
    y = convert_container(y, y_container)
    report = evaluate(LogisticRegression(max_iter=1000), X, y, splitter=3)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD005" in issues.index
    assert "Accuracy should not be used alone" in issues.loc["SKD005", "explanation"]

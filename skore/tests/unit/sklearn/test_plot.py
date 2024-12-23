import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skore.sklearn._plot import RocCurveDisplay


@pytest.fixture
def binary_classification_data():
    """Create a binary classification dataset and return fitted estimator and data."""
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return RandomForestClassifier().fit(X_train, y_train), X_test, y_test


def test_display_help(capsys, binary_classification_data):
    """Check that the help method writes to the console."""
    estimator, X, y = binary_classification_data
    display = RocCurveDisplay.from_estimator(estimator, X, y)

    display.help()
    captured = capsys.readouterr()
    assert f"📊 {display.__class__.__name__}" in captured.out


def test_display_repr(binary_classification_data):
    """Check that __repr__ returns a string starting with the expected prefix."""
    estimator, X, y = binary_classification_data
    display = RocCurveDisplay.from_estimator(estimator, X, y)

    repr_str = repr(display)
    assert repr_str.startswith(f"📊 {display.__class__.__name__}")


def test_roc_curve_display_plot(pyplot, binary_classification_data):
    """Only check the skore specific part of our specialized class."""
    estimator, X, y = binary_classification_data
    display = RocCurveDisplay.from_estimator(estimator, X, y)
    assert isinstance(display, RocCurveDisplay)
    assert isinstance(display.plot(), RocCurveDisplay)

    # check the despine works as expected
    display.plot(despine=True)
    for s in ["bottom", "left"]:
        assert display.ax_.spines[s].get_bounds() == (0, 1)

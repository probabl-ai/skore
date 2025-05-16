import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


@pytest.fixture
def regression_data():
    X, y = make_regression(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return LinearRegression().fit(X_train, y_train), X_train, X_test, y_train, y_test


@pytest.fixture
def regression_data_no_split():
    X, y = make_regression(random_state=42)
    return LinearRegression(), X, y

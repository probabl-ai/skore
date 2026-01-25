"""Unit tests for helpers in ``skore._utils.repr.data``."""

from unittest.mock import patch
from urllib.parse import quote

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from skore._utils._testing import MockDisplay, MockReport
from skore._utils.repr.data import (
    _build_attribute_text_fragment,
    _get_attribute_type,
    get_documentation_url,
)


class _ClassWithNumpydocAttrs:
    """Class with numpydoc-style Attributes section.

    Parameters
    ----------
    param : int
        Parameter.

    Attributes
    ----------
    coef_ : ndarray (n_features,)
        Coefficients.
    """


class _ClassWithNoDocstring:
    pass


@pytest.mark.parametrize(
    "obj, attribute_name, expected",
    [
        (_ClassWithNumpydocAttrs(), "coef_", "ndarray (n_features,)"),
        (_ClassWithNumpydocAttrs(), "param", "int"),
        (_ClassWithNoDocstring(), "coef_", None),
        (_ClassWithNumpydocAttrs(), "other_attr", None),
    ],
)
def test_get_attribute_type(obj, attribute_name, expected):
    """_get_attribute_type returns the type for numpydoc entries, else None."""
    assert _get_attribute_type(obj, attribute_name) == expected


@pytest.mark.parametrize(
    "obj, attribute_name, expected",
    [
        (_ClassWithNumpydocAttrs(), "coef_", "ndarray (n_features,)"),
        (_ClassWithNumpydocAttrs(), "param", "int"),
        (_ClassWithNoDocstring(), "coef_", None),
        (_ClassWithNumpydocAttrs(), "other_attr", None),
    ],
)
def test_build_attribute_text_fragment(obj, attribute_name, expected):
    """_build_attribute_text_fragment returns encoded name,-type or encoded name."""
    got = _build_attribute_text_fragment(obj, attribute_name)
    if expected is not None:
        expected = f"{quote(attribute_name, safe='')},-{quote(expected, safe='')}"
    else:
        expected = quote(attribute_name, safe="")
    assert got == expected


@pytest.fixture
def mock_report():
    """Minimal report instance for get_documentation_url tests."""
    X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y = np.array([0, 1, 1, 0])
    estimator = LogisticRegression().fit(X, y)
    return MockReport(estimator)


@pytest.fixture
def mock_display():
    """Minimal display instance for get_documentation_url tests."""
    return MockDisplay()


def test_get_documentation_url_class_only(mock_report, mock_display):
    """get_documentation_url returns base class URL with no accessor/method/attribute."""
    for obj in (mock_report, mock_display):
        url = get_documentation_url(obj=obj)
        assert url.startswith("https://docs.skore.probabl.ai/")
        assert "/reference/api/" in url
        assert "skore." in url and obj.__class__.__name__ in url
        assert url.endswith(".html")
        assert "#" not in url


def test_get_documentation_url_method_no_accessor(mock_report, mock_display):
    """get_documentation_url appends #skore.Class.method when method set, no accessor."""
    for obj in (mock_report, mock_display):
        url = get_documentation_url(obj=obj, method_name="help")
        assert url.startswith("https://docs.skore.probabl.ai/")
        assert ".html#" in url
        assert "#skore." in url and "help" in url


def test_get_documentation_url_accessor(mock_report):
    """get_documentation_url includes accessor in path when accessor_name set."""
    url = get_documentation_url(obj=mock_report, accessor_name="data")
    assert "skore.MockReport.data.html" in url
    assert "#" not in url


def test_get_documentation_url_accessor_and_method(mock_report):
    """get_documentation_url includes accessor and method in path."""
    url = get_documentation_url(
        obj=mock_report, accessor_name="data", method_name="get_data"
    )
    assert "skore.MockReport.data.get_data.html" in url
    assert "#" not in url


def test_get_documentation_url_attribute():
    """get_documentation_url appends #:~:text= fragment when attribute_name set."""
    obj = _ClassWithNumpydocAttrs()
    url = get_documentation_url(obj=obj, attribute_name="coef_")
    assert url.startswith("https://docs.skore.probabl.ai/")
    assert ".html#:~:text=" in url
    assert quote("coef_", safe="") in url


@pytest.mark.parametrize(
    "mocked_version, expected_url_version",
    [
        ("0.0.1", "dev"),
        ("0.0.0.dev0", "dev"),
        ("0.1.0", "0.1"),
        ("1.2.3", "1.2"),
    ],
)
def test_get_documentation_url_version_branches(
    mock_display, mocked_version, expected_url_version
):
    """get_documentation_url uses \"dev\" for version < 0.1, else major.minor."""
    with patch("skore._utils.repr.data.version", return_value=mocked_version):
        url = get_documentation_url(obj=mock_display)
    assert url.startswith("https://docs.skore.probabl.ai/")
    assert f"docs.skore.probabl.ai/{expected_url_version}/reference/api/" in url

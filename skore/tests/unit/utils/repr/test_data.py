"""Unit tests for helpers in ``skore._utils.repr.data``."""

import pytest

from skore._utils.repr.data import _get_attribute_type


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
    "obj_cls,attribute_name,expected",
    [
        (_ClassWithNumpydocAttrs, "coef_", "ndarray (n_features,)"),
        (_ClassWithNumpydocAttrs, "param", "int"),
        (_ClassWithNoDocstring, "coef_", None),
        (_ClassWithNumpydocAttrs, "other_attr", None),
    ],
)
def test_get_attribute_type(obj_cls, attribute_name, expected):
    """_get_attribute_type returns the type for numpydoc entries, else None."""
    obj = obj_cls()
    assert _get_attribute_type(obj, attribute_name) == expected

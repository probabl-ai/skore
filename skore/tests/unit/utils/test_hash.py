import numpy
import pytest
from skore._config import config_context
from skore.utils._hash import _hash


@pytest.mark.parametrize(
    "obj",
    [
        3,
        numpy.array([0.3]),
        (numpy.array([0.9]), numpy.array([0.3])),
    ],
)
def test_hash(obj):
    _hash(obj)


@pytest.mark.parametrize(
    "obj",
    [
        3,
        numpy.array([0.3]),
        (numpy.array([0.9]), numpy.array([0.3])),
    ],
)
def test_hash_different_config(obj):
    default = _hash(obj)

    with config_context(hash_func="sha1"):
        sha1 = _hash(obj)

    assert default != sha1

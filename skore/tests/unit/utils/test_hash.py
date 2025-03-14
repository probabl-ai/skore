import numpy
import pytest
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

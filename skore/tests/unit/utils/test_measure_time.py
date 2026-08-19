from skore.utils._measure_time import MeasureTime


def test():
    with MeasureTime() as t:
        assert 1 + 1 == 2
    assert isinstance(t(), float)

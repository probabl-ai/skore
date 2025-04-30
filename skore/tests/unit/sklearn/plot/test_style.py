import pytest
from skore.sklearn._plot.style import StyleDisplayMixin


class TestDisplay(StyleDisplayMixin):
    _default_some_kwargs = None


def test_style_mixin():
    display = TestDisplay()
    display.set_style(some_kwargs=1)
    assert display._default_some_kwargs == 1

    with pytest.raises(ValueError, match="Unknown style parameter: unknown_param."):
        display.set_style(unknown_param=1)

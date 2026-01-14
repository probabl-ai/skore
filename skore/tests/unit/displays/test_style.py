import pytest

from skore._sklearn._plot.base import StyleDisplayMixin


class TestDisplay(StyleDisplayMixin):
    _default_some_kwargs = None


def test_style_mixin():
    """Check that the style mixin works as intended."""
    display = TestDisplay()
    display.set_style(some_kwargs=1)
    assert display._default_some_kwargs == 1

    with pytest.raises(ValueError, match="Unknown style parameter: unknown_param."):
        display.set_style(unknown_param=1)


@pytest.mark.parametrize(
    "initial_state, override_value, expected_result, use_explicit_policy",
    [
        (None, {"a": 1, "b": 2}, {"a": 1, "b": 2}, False),
        ({"a": 1, "b": 2}, {"c": 3}, {"a": 1, "b": 2, "c": 3}, False),
        ({"c": 3}, {"a": 1, "b": 2}, {"a": 1, "b": 2}, True),
        ({"a": 1, "b": 2}, 42, 42, True),
    ],
)
def test_style_policy_override(
    initial_state, override_value, expected_result, use_explicit_policy
):
    """Check that the override policy replaces existing values."""
    display = TestDisplay()
    display._default_some_kwargs = initial_state

    if use_explicit_policy:
        display.set_style(some_kwargs=override_value, policy="override")
    else:
        display.set_style(some_kwargs=override_value)
    assert display._default_some_kwargs == expected_result


@pytest.mark.parametrize(
    "initial_state, update_value, expected_result",
    [
        (None, {"a": 1, "b": 2}, {"a": 1, "b": 2}),
        ({"a": 1, "b": 2}, {"b": 3, "c": 4}, {"a": 1, "b": 3, "c": 4}),
        (
            {"a": 1, "b": 3, "c": 4},
            {"a": 10, "d": 5},
            {"a": 10, "b": 3, "c": 4, "d": 5},
        ),
    ],
)
def test_style_policy_update(initial_state, update_value, expected_result):
    """Check that the update policy merges dictionaries or sets if None."""
    display = TestDisplay()
    display._default_some_kwargs = initial_state

    display.set_style(some_kwargs=update_value, policy="update")
    assert display._default_some_kwargs == expected_result


def test_style_policy_invalid():
    """Check that invalid policy raises ValueError."""
    display = TestDisplay()

    err_msg = (
        "Invalid policy: invalid_policy. Valid policies are 'override' and 'update'."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.set_style(some_kwargs=1, policy="invalid_policy")

import pytest

from skore._displays.base import StyleDisplayMixin


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
        (
            {
                "facet_kws": {
                    "sharex": False,
                    "sharey": False,
                    "xlim": (-0.01, 1.01),
                    "ylim": (-0.01, 1.01),
                }
            },
            {"facet_kws": {"xlim": (0, 1)}},
            {
                "facet_kws": {
                    "sharex": False,
                    "sharey": False,
                    "xlim": (0, 1),
                    "ylim": (-0.01, 1.01),
                }
            },
        ),
    ],
)
def test_style_policy_update(initial_state, update_value, expected_result):
    """Check that update replaces passed keys and keeps omitted nested keys."""
    display = TestDisplay()
    display._default_some_kwargs = initial_state

    display.set_style(some_kwargs=update_value, policy="update")
    assert display._default_some_kwargs == expected_result


def test_style_policy_update_does_not_mutate_nested_input():
    """Updating a nested mapping must not rewrite the caller's dictionary."""
    original = {
        "facet_kws": {
            "sharex": False,
            "xlim": (-0.01, 1.01),
            "ylim": (-0.01, 1.01),
        }
    }
    display = TestDisplay()
    display._default_some_kwargs = original

    display.set_style(some_kwargs={"facet_kws": {"xlim": (0, 1)}}, policy="update")

    assert original["facet_kws"]["xlim"] == (-0.01, 1.01)
    assert display._default_some_kwargs["facet_kws"]["sharex"] is False
    assert display._default_some_kwargs["facet_kws"]["xlim"] == (0, 1)
    assert display._default_some_kwargs["facet_kws"]["ylim"] == (-0.01, 1.01)


def test_roc_set_style_merges_facet_kws():
    """ROC facet limits stay in place when only one limit is updated."""
    from skore._displays.metrics.roc_curve import RocCurveDisplay

    display = RocCurveDisplay.__new__(RocCurveDisplay)
    display.set_style(relplot_kwargs={"facet_kws": {"xlim": (0, 1)}})

    facet_kws = display._default_relplot_kwargs["facet_kws"]
    assert facet_kws["xlim"] == (0, 1)
    assert facet_kws["ylim"] == (-0.01, 1.01)
    assert facet_kws["sharex"] is False
    assert facet_kws["sharey"] is False
    assert display._default_chance_level_kwargs == {"color": "k", "linestyle": "--"}
    assert RocCurveDisplay._default_relplot_kwargs["facet_kws"]["xlim"] == (-0.01, 1.01)


def test_style_policy_invalid():
    """Check that invalid policy raises ValueError."""
    display = TestDisplay()

    err_msg = (
        "Invalid policy: invalid_policy. Valid policies are 'override' and 'update'."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.set_style(some_kwargs=1, policy="invalid_policy")

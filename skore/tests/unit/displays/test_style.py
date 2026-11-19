from collections import UserDict

import pytest

from skore._displays.base import StyleDisplayMixin
from skore._displays.data.table_report import TableReportDisplay
from skore._displays.inspection.calibration_curve import CalibrationDisplay
from skore._displays.inspection.coefficients import CoefficientsDisplay
from skore._displays.inspection.impurity_decrease import ImpurityDecreaseDisplay
from skore._displays.inspection.permutation_importance import (
    PermutationImportanceDisplay,
)
from skore._displays.metrics.confusion_matrix import ConfusionMatrixDisplay
from skore._displays.metrics.precision_recall_curve import PrecisionRecallCurveDisplay
from skore._displays.metrics.prediction_error import PredictionErrorDisplay
from skore._displays.metrics.roc_curve import RocCurveDisplay


class TestDisplay(StyleDisplayMixin):
    _default_some_kwargs = None


class MultiGroupDisplay(StyleDisplayMixin):
    _default_curve_kwargs = {"color": "C0"}
    _default_line_kwargs = {"color": "k", "linestyle": "--"}


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


def test_style_policy_update_merges_non_dict_mapping():
    """A stored mapping that is not a dict is shallow-merged on update."""
    display = TestDisplay()
    display._default_some_kwargs = UserDict({"a": 1, "b": 2})

    display.set_style(some_kwargs={"b": 3, "c": 4}, policy="update")

    assert display._default_some_kwargs == {"a": 1, "b": 3, "c": 4}


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


def test_override_leaves_omitted_groups():
    """Override replaces only the groups that were passed."""
    display = MultiGroupDisplay()

    display.set_style(curve_kwargs={"color": "tab:red"}, policy="override")

    assert display._default_curve_kwargs == {"color": "tab:red"}
    assert display._default_line_kwargs == {"color": "k", "linestyle": "--"}


def test_override_empty_dict_clears_only_that_group():
    """An explicit empty dict clears that group and no other."""
    display = MultiGroupDisplay()

    display.set_style(curve_kwargs={}, policy="override")

    assert display._default_curve_kwargs == {}
    assert display._default_line_kwargs == {"color": "k", "linestyle": "--"}


@pytest.mark.parametrize(
    "display_cls, passed, untouched",
    [
        (RocCurveDisplay, "relplot_kwargs", "_default_chance_level_kwargs"),
        (PredictionErrorDisplay, "relplot_kwargs", "_default_perfect_model_kwargs"),
        (ConfusionMatrixDisplay, "heatmap_kwargs", "_default_facet_grid_kwargs"),
        (
            PermutationImportanceDisplay,
            "boxplot_kwargs",
            "_default_stripplot_kwargs",
        ),
        (CoefficientsDisplay, "barplot_kwargs", "_default_stripplot_kwargs"),
        (ImpurityDecreaseDisplay, "barplot_kwargs", "_default_boxplot_kwargs"),
        (CalibrationDisplay, "line_kwargs", "_default_ax_set_kwargs"),
        (TableReportDisplay, "histplot_kwargs", "_default_scatterplot_kwargs"),
        (PrecisionRecallCurveDisplay, "relplot_kwargs", None),
    ],
)
def test_display_override_keeps_omitted_style_groups(display_cls, passed, untouched):
    """Display wrappers must not turn omitted arguments into an empty override."""
    display = display_cls.__new__(display_cls)
    before = None if untouched is None else dict(getattr(display, untouched))

    display.set_style(**{passed: {"color": "tab:red"}}, policy="override")

    assert getattr(display, f"_default_{passed}") == {"color": "tab:red"}
    if untouched is not None:
        assert untouched not in display.__dict__
        assert getattr(display, untouched) == before

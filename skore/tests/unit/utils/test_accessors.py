import pytest
from sklearn.pipeline import make_pipeline

from skore._externals.pandas_accessors import DirNamesMixin, _register_accessor
from skore._utils.accessor import (
    _check_all_checks,
    _check_comparison_report_sub_estimators_have_coef,
    _check_comparison_report_sub_estimators_have_feature_importances,
    _check_cross_validation_sub_estimator_has_coef,
    _check_estimator_has_coef,
    _check_estimator_has_feature_importances,
    _check_estimator_report_has_method,
    _check_has_coef,
    _check_roc_auc,
    _check_supported_ml_task,
    _expand_data_sources,
)


def test_register_accessor():
    """Test that an accessor is properly registered and accessible on a class
    instance.
    """

    class ParentClass(DirNamesMixin):
        pass

    def register_parent_class_accessor(name: str):
        """Register an accessor for the ParentClass class."""
        return _register_accessor(name, ParentClass)

    @register_parent_class_accessor("accessor")
    class _Accessor:
        def __init__(self, parent):
            self._parent = parent

        def func(self):
            return True

    obj = ParentClass()
    assert hasattr(obj, "accessor")
    assert isinstance(obj.accessor, _Accessor)
    assert obj.accessor.func()


def test_check_supported_ml_task():
    """Test that ML task validation accepts supported tasks and rejects unknown
    ones.
    """

    class MockParent:
        def __init__(self, ml_task):
            self._ml_task = ml_task

    class MockAccessor:
        def __init__(self, parent):
            self._parent = parent

    parent = MockParent("binary-classification")
    accessor = MockAccessor(parent)
    check = _check_supported_ml_task(
        ["binary-classification", "multiclass-classification"]
    )
    assert check(accessor)

    parent = MockParent("multiclass-classification")
    accessor = MockAccessor(parent)
    assert check(accessor)

    parent = MockParent("regression")
    accessor = MockAccessor(parent)
    err_msg = "The regression task is not a supported task by function called."
    with pytest.raises(AttributeError, match=err_msg):
        check(accessor)


def test_check_estimator_has_coef():
    """
    Test that only estimators with the `coef_` attribute are accepted.
    """

    class MockParent:
        def __init__(self, estimator):
            self.estimator_ = estimator

    class MockAccessor:
        def __init__(self, parent):
            self._parent = parent

    class Estimator:
        def __init__(self):
            self.coef_ = 0

    class MetaEstimator:
        def __init__(self):
            self.regressor_ = Estimator()

    parent = MockParent(Estimator())
    accessor = MockAccessor(parent)

    assert _check_estimator_has_coef()(accessor)

    parent = MockParent(make_pipeline(Estimator()))
    accessor = MockAccessor(parent)

    assert _check_estimator_has_coef()(accessor)

    parent = MockParent(MetaEstimator())
    accessor = MockAccessor(parent)

    assert _check_estimator_has_coef()(accessor)

    parent = MockParent(estimator="hello")
    accessor = MockAccessor(parent)

    err_msg = "Estimator 'hello' is not a supported estimator by the function called."
    with pytest.raises(AttributeError, match=err_msg):
        assert _check_estimator_has_coef()(accessor)


def test_check_estimator_report_has_coef():
    """
    Test that `CrossValidationReport` only allows access to estimators that expose a
    `coef_` attribute.
    """

    class Estimator:
        def __init__(self):
            self.coef_ = 0

    class MetaEstimator:
        def __init__(self):
            self.regressor_ = Estimator()

    class MockReport:
        def __init__(self, estimator):
            self.estimator_ = estimator

    class MockParent:
        def __init__(self, estimator):
            self.reports_ = [MockReport(estimator)]

    class MockAccessor:
        def __init__(self, parent):
            self._parent = parent

    accessor = MockAccessor(MockParent(Estimator()))
    assert _check_cross_validation_sub_estimator_has_coef()(accessor)

    accessor = MockAccessor(MockParent(make_pipeline(Estimator())))
    assert _check_cross_validation_sub_estimator_has_coef()(accessor)

    accessor = MockAccessor(MockParent(MetaEstimator()))
    assert _check_cross_validation_sub_estimator_has_coef()(accessor)

    accessor = MockAccessor(MockParent("hello"))
    err_msg = "Estimator 'hello' is not a supported estimator by the function called."
    with pytest.raises(AttributeError, match=err_msg):
        _check_cross_validation_sub_estimator_has_coef()(accessor)


def test_check_has_feature_importance():
    """
    Test that only estimators with the `feature_importances_` attribute are accepted.
    """

    class MockParent:
        def __init__(self, estimator):
            self.estimator_ = estimator

    class MockAccessor:
        def __init__(self, parent):
            self._parent = parent

    class Estimator:
        def __init__(self):
            self.feature_importances_ = 0

    parent = MockParent(Estimator())
    accessor = MockAccessor(parent)

    assert _check_estimator_has_feature_importances()(accessor)

    parent = MockParent(make_pipeline(Estimator()))
    accessor = MockAccessor(parent)

    assert _check_estimator_has_feature_importances()(accessor)

    parent = MockParent(estimator="hello")
    accessor = MockAccessor(parent)

    err_msg = "Estimator 'hello' is not a supported estimator by the function called."
    with pytest.raises(AttributeError, match=err_msg):
        assert _check_estimator_has_feature_importances()(accessor)


def test_check_all_checks():
    """A composite check passes only when every nested check passes."""
    assert _check_all_checks([lambda accessor: True, lambda accessor: True])(object())
    assert not _check_all_checks([lambda accessor: True, lambda accessor: False])(
        object()
    )


def test_check_has_coef_without_regressor_coefficient():
    """A nested regressor without ``coef_`` is rejected."""

    class MetaEstimator:
        def __init__(self):
            self.regressor_ = object()

    with pytest.raises(AttributeError, match="not a supported estimator"):
        _check_has_coef(MetaEstimator())


def test_check_has_coef_reraises_unexpected_attribute_error():
    """Attribute errors other than a missing regressor propagate."""

    class Estimator:
        @property
        def regressor_(self):
            raise AttributeError("custom failure")

    with pytest.raises(AttributeError, match="custom failure"):
        _check_has_coef(Estimator())


def test_check_roc_auc():
    """ROC-AUC support requires one accepted task and prediction method."""

    class Learner:
        def predict_proba(self):
            return None

    class Parent:
        def __init__(self, ml_task, learner):
            self._ml_task = ml_task
            self.learner_ = learner

    class Accessor:
        def __init__(self, parent):
            self._parent = parent

    check = _check_roc_auc(
        [
            ("binary-classification", ["predict_proba"]),
            ("multiclass-classification", ["decision_function"]),
        ]
    )
    accessor = Accessor(Parent("binary-classification", Learner()))
    assert check(accessor)

    accessor = Accessor(Parent("regression", object()))
    with pytest.raises(AttributeError, match="right prediction methods"):
        check(accessor)


def test_expand_data_sources():
    """``both`` expands to the train and test sources."""
    assert _expand_data_sources("both") == ("train", "test")
    assert _expand_data_sources("train") == ("train",)


def test_estimator_report_check_requires_accessor():
    """Cross-validation checks fail when the sub-report has no accessor."""

    class Parent:
        reports_ = [object()]

    class Accessor:
        def __init__(self, parent):
            self._parent = parent

    check = _check_estimator_report_has_method("metrics", "roc")
    with pytest.raises(AttributeError, match="does not have the 'metrics' accessor"):
        check(Accessor(Parent()))


@pytest.mark.parametrize(
    "check_factory",
    [
        _check_comparison_report_sub_estimators_have_coef,
        _check_comparison_report_sub_estimators_have_feature_importances,
    ],
)
def test_comparison_checks_reject_unexpected_report_type(check_factory):
    """Comparison coefficient checks only accept the two comparison report types."""

    class Parent:
        _report_type = "estimator"
        reports_ = [object()]

    class Accessor:
        def __init__(self, parent):
            self._parent = parent

    with pytest.raises(TypeError, match="Unexpected report type"):
        check_factory()(Accessor(Parent()))

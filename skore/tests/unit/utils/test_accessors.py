import pytest
from sklearn.pipeline import make_pipeline
from skore._externals._pandas_accessors import DirNamesMixin, _register_accessor
from skore._utils._accessor import (
    _check_estimator_report_has_coef,
    _check_has_coef,
    _check_has_feature_importances,
    _check_supported_ml_task,
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


def test_check_has_coef():
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

    assert _check_has_coef()(accessor)

    parent = MockParent(make_pipeline(Estimator()))
    accessor = MockAccessor(parent)

    assert _check_has_coef()(accessor)

    parent = MockParent(MetaEstimator())
    accessor = MockAccessor(parent)

    assert _check_has_coef()(accessor)

    parent = MockParent(estimator="hello")
    accessor = MockAccessor(parent)

    err_msg = "Estimator 'hello' is not a supported estimator by the function called."
    with pytest.raises(AttributeError, match=err_msg):
        assert _check_has_coef()(accessor)


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
            self.estimator = estimator

    class MockParent:
        def __init__(self, estimator):
            self.estimator_reports_ = [MockReport(estimator)]

    class MockAccessor:
        def __init__(self, parent):
            self._parent = parent

    accessor = MockAccessor(MockParent(Estimator()))
    assert _check_estimator_report_has_coef()(accessor)

    accessor = MockAccessor(MockParent(make_pipeline(Estimator())))
    assert _check_estimator_report_has_coef()(accessor)

    accessor = MockAccessor(MockParent(MetaEstimator()))
    assert _check_estimator_report_has_coef()(accessor)

    accessor = MockAccessor(MockParent("hello"))
    err_msg = "Estimator 'hello' is not a supported estimator by the function called."
    with pytest.raises(AttributeError, match=err_msg):
        _check_estimator_report_has_coef()(accessor)


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

    assert _check_has_feature_importances()(accessor)

    parent = MockParent(make_pipeline(Estimator()))
    accessor = MockAccessor(parent)

    assert _check_has_feature_importances()(accessor)

    parent = MockParent(estimator="hello")
    accessor = MockAccessor(parent)

    err_msg = "Estimator 'hello' is not a supported estimator by the function called."
    with pytest.raises(AttributeError, match=err_msg):
        assert _check_has_feature_importances()(accessor)

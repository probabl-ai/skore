import pytest
from skore.utils._accessor import (
    _check_supported_ml_task,
    register_accessor,
)


def test_register_accessor():
    """Test that an accessor is properly registered and accessible on a class
    instance.
    """

    class ParentClass:
        pass

    @register_accessor("accessor", ParentClass)
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
    """Test that ML task validation accepts supported tasks and rejects unsupported
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

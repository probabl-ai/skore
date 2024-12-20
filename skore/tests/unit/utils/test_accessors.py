from skore.utils._accessor import register_accessor


def test_register_accessor():
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

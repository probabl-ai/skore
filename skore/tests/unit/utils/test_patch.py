import math

from skore._utils._patch import patch_function, patch_instance_method


def test_patch_function_basic():
    """Test that patch_function temporarily replaces a module function."""
    original_sqrt = math.sqrt

    def patched_sqrt(x):
        return x * 2

    with patch_function(math, "sqrt", patched_sqrt):
        assert math.sqrt(4) == 8

    assert math.sqrt(4) == 2.0
    assert math.sqrt is original_sqrt


def test_patch_function_restores_on_exception():
    """Test that patch_function restores original function even if exception occurs."""
    original_sqrt = math.sqrt

    def patched_sqrt(x):
        return x * 2

    try:
        with patch_function(math, "sqrt", patched_sqrt):
            assert math.sqrt(4) == 8
            raise ValueError("Test exception")
    except ValueError:
        pass

    assert math.sqrt(4) == 2.0
    assert math.sqrt is original_sqrt


def test_patch_function_nonexistent_function():
    """Test that patch_function handles non-existent functions gracefully."""

    with patch_function(math, "nonexistent_function", lambda x: x):
        pass

    assert not hasattr(math, "nonexistent_function")


def test_patch_instance_method_basic():
    """Test that patch_instance_method patches a method on a specific instance."""

    class MyClass:
        def greet(self):
            return "Hello"

    obj1 = MyClass()
    obj2 = MyClass()

    def patched_greet(self):
        return "Hola"

    with patch_instance_method(obj1, "greet", patched_greet):
        assert obj1.greet() == "Hola"
        assert obj2.greet() == "Hello"

    assert obj1.greet() == "Hello"
    assert obj2.greet() == "Hello"


def test_patch_instance_method_restores_on_exception():
    """Test that patch_instance_method restores original method even if exception
    occurs."""

    class MyClass:
        def greet(self):
            return "Hello"

    obj = MyClass()

    def patched_greet(self):
        return "Hola"

    try:
        with patch_instance_method(obj, "greet", patched_greet):
            assert obj.greet() == "Hola"
            raise ValueError("Test exception")
    except ValueError:
        pass

    assert obj.greet() == "Hello"


def test_patch_instance_method_nonexistent_method():
    """Test that patch_instance_method handles non-existent methods gracefully."""

    class MyClass:
        pass

    obj = MyClass()

    with patch_instance_method(obj, "nonexistent_method", lambda self: "test"):
        pass

    assert not hasattr(obj, "nonexistent_method")


def test_patch_instance_method_multiple_instances_unaffected():
    """Test that patching one instance doesn't affect other instances."""

    class MyClass:
        def value(self):
            return 10

    obj1 = MyClass()
    obj2 = MyClass()
    obj3 = MyClass()

    def patched_value(self):
        return 99

    with patch_instance_method(obj2, "value", patched_value):
        assert obj1.value() == 10
        assert obj2.value() == 99
        assert obj3.value() == 10

    # All should be restored
    assert obj1.value() == 10
    assert obj2.value() == 10
    assert obj3.value() == 10

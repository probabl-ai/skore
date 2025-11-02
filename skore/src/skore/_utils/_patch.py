import os
import types
from collections.abc import Iterable
from contextlib import contextmanager

from rich import jupyter
from rich.segment import Segment

_original_render_segments = jupyter._render_segments


def patched_render_segments(segments: Iterable[Segment]) -> str:
    """Patched version of rich.jupyter._render_segments that includes VS Code styling.

    This is to make sure that the CSS style exposed by jupyter notebook and used
    by ipywidgets is applied to rich output.
    Currently, the VS Code extension does not support the CSS style exposed by jupyter
    notebook and used by ipywidgets.
    We should track the progress in the following issue:
    https://github.com/microsoft/vscode-jupyter/issues/7161

    """
    html = _original_render_segments(segments)
    # Apply VS Code styling if we're in VS Code
    if "VSCODE_PID" in os.environ:
        css = """
        <style>
        .cell-output-ipywidget-background {
            background-color: transparent !important;
        }
        :root {
            --jp-widgets-color: var(--vscode-editor-foreground);
            --jp-widgets-font-size: var(--vscode-editor-font-size);
        }
        </style>
        """
        html += css

    return html


def setup_jupyter_display() -> None:
    """Configure the jupyter display to work properly in VS Code."""
    jupyter._render_segments = patched_render_segments


@contextmanager
def patch_function(module, function_name, new_function):
    """Temporarily patch a module-level function.

    Parameters
    ----------
    module : module
        The module containing the function.
    function_name : str
        Name of the function to patch.
    new_function : callable
        The replacement function.

    Yields
    ------
    None
        This context manager yields nothing.

    Notes
    -----
    If the function does not exist in the module, this context manager
    has no effect and does not raise an error.

    Examples
    --------
    >>> import math
    >>> with patch_function(math, 'sqrt', lambda x: x ** 0.5):
    ...     result = math.sqrt(16)  # Uses patched version
    """
    if not hasattr(module, function_name):
        yield
        return

    original = getattr(module, function_name)
    setattr(module, function_name, new_function)

    try:
        yield
    finally:
        setattr(module, function_name, original)


@contextmanager
def patch_instance_method(instance, method_name, new_method):
    """Temporarily patch a method on a specific instance (only affects this instance).

    Parameters
    ----------
    instance : object
        The object instance to patch.
    method_name : str
        Name of the method to patch.
    new_method : callable
        The replacement method (should accept self as first argument).

    Yields
    ------
    None
        This context manager yields nothing.

    Notes
    -----
    If the method does not exist on the instance or its class, this context
    manager has no effect and does not raise an error.

    Examples
    --------
    >>> class MyClass:
    ...     def greet(self):
    ...         return "Hello"
    >>> obj1 = MyClass()
    >>> obj2 = MyClass()
    >>> with patch_instance_method(obj1, 'greet', lambda self: "Hola"):
    ...     obj1.greet()  # Returns "Hola"
    ...     obj2.greet()  # Returns "Hello" (unaffected)
    """
    if not hasattr(instance, method_name):
        yield
        return

    has_instance_method = method_name in instance.__dict__
    original = getattr(instance, method_name) if has_instance_method else None

    bound_method = types.MethodType(new_method, instance)
    setattr(instance, method_name, bound_method)

    try:
        yield
    finally:
        if has_instance_method:
            setattr(instance, method_name, original)
        else:
            delattr(instance, method_name)

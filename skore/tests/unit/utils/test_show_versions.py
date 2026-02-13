from skore._utils._show_versions import _get_deps_info, _get_sys_info, show_versions


def test_get_sys_info():
    sys_info = _get_sys_info()
    assert isinstance(sys_info, dict)
    assert "python" in sys_info
    assert "executable" in sys_info
    assert "machine" in sys_info


def test_get_deps_info():
    deps_info = _get_deps_info()
    assert isinstance(deps_info, dict)
    assert "pip" in deps_info
    assert "skore" in deps_info


def test_show_versions(capfd):
    """Check that we have the expected packages in the output of `show_versions()`.

    We use `:` in the assertion to be sure that we are robust to package
    version specifiers.

    Non-regression test for:
    https://github.com/probabl-ai/skore/issues/987
    """
    show_versions()
    captured = capfd.readouterr()
    assert "python:" in captured.out
    assert "executable:" in captured.out
    assert "machine:" in captured.out
    assert "skore:" in captured.out
    assert "pip:" in captured.out
    assert "numpy:" in captured.out
    assert "rich[jupyter]:" in captured.out
    assert "scikit-learn:" in captured.out

import os
import tempfile

from skore.utils._show_versions import _get_deps_info, _get_sys_info, show_versions


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
    assert "setuptools" in deps_info
    assert "skore" in deps_info


def test_show_versions(capfd):
    show_versions()
    captured = capfd.readouterr()
    assert "python" in captured.out
    assert "executable" in captured.out
    assert "machine" in captured.out
    assert "pip" in captured.out
    assert "setuptools" in captured.out
    assert "skore" in captured.out


def test_get_deps_in_any_working_directory(capfd):
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        show_versions()
        captured = capfd.readouterr()
        assert "python" in captured.out
        assert "executable" in captured.out
        assert "machine" in captured.out
        assert "pip" in captured.out
        assert "setuptools" in captured.out
        assert "skore" in captured.out
    os.chdir(cwd)

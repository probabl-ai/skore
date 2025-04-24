from skore_remote_project.item.item import (
    b64_str_to_bytes,
    bytes_to_b64_str,
    lazy_is_instance,
)


def test_lazy_is_instance():
    assert lazy_is_instance(1, "builtins.int")
    assert lazy_is_instance(1, "builtins.object")
    assert not lazy_is_instance(1, "builtins.float")
    assert lazy_is_instance(tuple(), "builtins.tuple")
    assert lazy_is_instance(tuple(), "builtins.object")
    assert not lazy_is_instance(tuple(), "builtins.dict")


def test_bytes_to_b64_str():
    assert bytes_to_b64_str(b"<str>") == "PHN0cj4="


def test_b64_str_to_bytes():
    assert b64_str_to_bytes("PHN0cj4=") == b"<str>"

"""Pandas-like accessors.

This code is copied from:
https://github.com/pandas-dev/pandas/blob/main/pandas/core/accessor.py

It is used to register accessors for the skore classes.
"""

from typing import final


class DirNamesMixin:
    _accessors: set[str] = set()
    _hidden_attrs: frozenset[str] = frozenset()

    @final
    def _dir_deletions(self) -> set[str]:
        return self._accessors | self._hidden_attrs

    def _dir_additions(self) -> set[str]:
        return {accessor for accessor in self._accessors if hasattr(self, accessor)}

    def __dir__(self) -> list[str]:
        rv = set(super().__dir__())
        rv = (rv - self._dir_deletions()) | self._dir_additions()
        return sorted(rv)


class Accessor:
    def __init__(self, name: str, accessor) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            # we're accessing the attribute of the class, i.e., Dataset.geo
            return self._accessor
        return self._accessor(obj)


def _register_accessor(name, cls):
    def decorator(accessor):
        if hasattr(cls, name):
            raise ValueError(
                f"registration of accessor {accessor!r} under name "
                f"{name!r} for type {cls!r} is overriding a preexisting "
                f"attribute with the same name."
            )
        setattr(cls, name, Accessor(name, accessor))
        cls._accessors.add(name)
        return accessor

    return decorator

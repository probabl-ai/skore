from collections import UserDict
from functools import wraps
from threading import RLock
from typing import Any


def method_with_explicit_lock(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "__lock__"):
            self.__lock__ = RLock()

        with self.__lock__:
            return method(self, *args, **kwargs)

    return wrapper


class Cache(UserDict[tuple[Any, ...], Any]):
    """Thread-safe cache based on ``dict``, with an explicit lock on write/delete."""

    __setitem__ = method_with_explicit_lock(UserDict.__setitem__)
    __delitem__ = method_with_explicit_lock(UserDict.__delitem__)
    clear = method_with_explicit_lock(UserDict.clear)
    pop = method_with_explicit_lock(UserDict.pop)
    popitem = method_with_explicit_lock(UserDict.popitem)
    update = method_with_explicit_lock(UserDict.update)

    def __getstate__(self):
        return self.data.copy()

    def __setstate__(self, state):
        self.data = state

from collections import UserDict
from functools import wraps
from threading import RLock
from typing import Any


def method_with_explicit_lock(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        self.__lock__.acquire()

        try:
            return method(self, *args, **kwargs)
        finally:
            self.__lock__.release()

    return wrapper


def method_generator_with_explicit_lock(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        self.__lock__.acquire()

        try:
            yield from method(self, *args, **kwargs)
        finally:
            self.__lock__.release()

    return wrapper


class Cache(UserDict[tuple[Any, ...], Any]):
    """Thread-safe cache based on ``dict``, with a lock on write/delete/iter."""

    __delitem__ = method_with_explicit_lock(UserDict.__delitem__)
    __iter__ = method_generator_with_explicit_lock(UserDict.__iter__)
    __setitem__ = method_with_explicit_lock(UserDict.__setitem__)
    clear = method_with_explicit_lock(UserDict.clear)
    pop = method_with_explicit_lock(UserDict.pop)
    popitem = method_with_explicit_lock(UserDict.popitem)
    update = method_with_explicit_lock(UserDict.update)

    def __init__(self, *args, **kwargs):
        self.__lock__ = RLock()
        super().__init__(*args, **kwargs)

    def __getstate__(self):
        return self.data.copy()

    def __setstate__(self, state):
        if not hasattr(self, "__lock__"):
            self.__lock__ = RLock()

        self.data = state

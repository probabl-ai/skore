from functools import partialmethod
from io import BytesIO

from joblib import dump, load
from pytest import mark, param, raises

from skore._utils._cache import Cache


@mark.parametrize(
    "input,output,action",
    (
        param(
            {},
            {0: 0},
            lambda cache: type(cache).__setitem__(cache, 0, 0),
            id="__setitem__",
        ),
        param(
            {0: 0, 1: 1},
            {0: 0},
            lambda cache: type(cache).__delitem__(cache, 1),
            id="__delitem__",
        ),
        param({0: 0, 1: 1}, {}, lambda cache: cache.clear(), id="clear"),
        param({0: 0, 1: 1}, {0: 0}, lambda cache: cache.pop(1), id="pop"),
        param({0: 0, 1: 1}, {1: 1}, lambda cache: cache.popitem(), id="popitem"),
        param({0: 0}, {0: 1}, lambda cache: cache.update({0: 1}), id="update"),
    ),
)
def test_cache_method_with_explicit_lock(monkeypatch, input, output, action):
    cache = Cache()
    cache.data = input

    assert not hasattr(cache, "__lock__")

    action(cache)

    assert hasattr(cache, "__lock__")
    assert cache == output


def test_cache_lockable(monkeypatch):
    # Ensure that cache uses the lock system properly.
    #
    # First, we patch the lock acquisition so that it timeouts when it has already been
    # acquired by another thread. We then manually lock the cache before attempting to
    # perform a `pop` operation. This should trigger an exception indicating that the
    # lock cannot be set.
    import concurrent.futures
    import threading

    monkeypatch.setattr("threading._CRLock", None)
    monkeypatch.setattr(
        "threading._PyRLock.__enter__",
        partialmethod(threading._PyRLock.__enter__, timeout=0),
    )

    cache = Cache({0: 0})

    with (
        cache.__lock__,
        concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool,
        raises(RuntimeError, match="cannot release un-acquired lock"),
    ):
        task = pool.submit((lambda cache: cache.pop(1)), cache)

        while task.running():
            pass

        task.result()


def test_cache_picklable():
    cache = Cache({0: 0})

    with BytesIO() as stream:
        dump(cache, stream)
        pickle = stream.getvalue()

    with BytesIO(pickle) as stream:
        unpickle = load(stream)

    assert cache == unpickle

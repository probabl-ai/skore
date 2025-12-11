from io import BytesIO

from joblib import dump, load
from pytest import mark, param, raises

from skore._utils._cache import Cache


@mark.parametrize(
    "input,output,state,method",
    (
        param(
            {0: 0, 1: 1},
            None,
            {0: 0},
            lambda cache: type(cache).__delitem__(cache, 1),
            id="__delitem__",
        ),
        param(
            {0: 0},
            [0],
            {0: 0},
            lambda cache: list(cache),
            id="__iter__",
        ),
        param(
            {},
            None,
            {0: 0},
            lambda cache: type(cache).__setitem__(cache, 0, 0),
            id="__setitem__",
        ),
        param(
            {0: 0, 1: 1},
            None,
            {},
            lambda cache: cache.clear(),
            id="clear",
        ),
        param(
            {0: 0, 1: 1},
            1,
            {0: 0},
            lambda cache: cache.pop(1),
            id="pop",
        ),
        param(
            {0: 0, 1: 1},
            (0, 0),
            {1: 1},
            lambda cache: cache.popitem(),
            id="popitem",
        ),
        param(
            {0: 0},
            None,
            {0: 1},
            lambda cache: cache.update({0: 1}),
            id="update",
        ),
    ),
)
def test_cache_method_is_functional(monkeypatch, input, output, state, method):
    cache = Cache(input)

    assert method(cache) == output
    assert cache == state


@mark.parametrize(
    "method",
    (
        param((lambda cache: cache.__delitem__()), id="__delitem__"),
        # param((lambda cache: list(cache.__iter__())), id="__iter__"),
        # param((lambda cache: cache.__setitem__()), id="__setitem__"),
        # param((lambda cache: cache.clear()), id="clear"),
        # param((lambda cache: cache.pop()), id="pop"),
        # param((lambda cache: cache.popitem()), id="popitem"),
        # param((lambda cache: cache.update()), id="update"),
    ),
)
def test_cache_method_is_explicitly_locked(monkeypatch, method):
    # Ensure that cache uses the lock system properly.
    #
    # First, we patch the lock acquisition so that it raises an exception when it has
    # already been acquired by another thread. We then manually lock the cache before
    # attempting to perform an operation. This should trigger an exception indicating
    # that the lock cannot be set.
    import concurrent.futures
    import threading

    def __enter__(self):
        if self.locked():
            raise RuntimeError("Lock already acquired")

        return self.acquire()

    monkeypatch.setattr("threading._CRLock", None)
    monkeypatch.setattr("threading._PyRLock.__enter__", __enter__)

    event = threading.Event()
    cache = Cache()

    def acquire_and_wait():
        cache.__lock__.acquire()

        if not event.wait(timeout=2):
            raise RuntimeError("An issue occurs during the test: timeout expired")

    # monkeypatch.setattr(cache.__lock__, "__enter__", __enter__)

    with (
        concurrent.futures.ThreadPoolExecutor() as pool,
        raises(RuntimeError, match="Lock already acquired"),
    ):
        task1 = pool.submit(acquire_and_wait)
        task2 = pool.submit(method, cache)

        task2.result()
        event.set()
        task1.result()


def test_cache_picklable():
    cache = Cache({0: 0})

    with BytesIO() as stream:
        dump(cache, stream)
        pickle = stream.getvalue()

    with BytesIO(pickle) as stream:
        unpickle = load(stream)

    assert cache == unpickle

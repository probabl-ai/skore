from typing import Any

from skore.externals._joblib_hash import hash

# A hashlib hash object, that has a `update` and a `digest` method
HashObject = Any


def _hash(obj, coerce_mmap=False):
    from skore import get_config

    return hash(obj, hash_name=get_config()["hash_func"], coerce_mmap=coerce_mmap)

from typing import Any

from sklearn.utils import murmurhash3_32 as sklearn_murmurhash3_32

from skore.externals._joblib_hash import hash

# A hashlib hash object, that has a `update` and a `digest` method
HashObject = Any


class murmurhash3_32:
    def __init__(self):
        self._hash = lambda obj: sklearn_murmurhash3_32(obj, positive=True)
        self.data = 0

    def update(self, data):
        self.data = (self.data + self._hash(bytes(data))) % (2**32)

    def digest(self):
        return int.to_bytes(self.data, length=4)

    def hexdigest(self):
        return self.digest().hex()

    @property
    def name(self):
        return "murmurhash3_32"


def _hash(obj, coerce_mmap=False):
    from skore import get_config

    return hash(obj, hash_name=get_config()["hash_func"], coerce_mmap=coerce_mmap)

from skore.externals._joblib_hash import hash


def _hash(obj, coerce_mmap=False):
    return hash(obj, hash_name="md5", coerce_mmap=coerce_mmap)

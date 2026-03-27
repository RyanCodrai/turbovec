import hashlib
import os
import pickle
from functools import wraps

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


def disk_cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        cache_bytes = f"{func.__name__}:{args}:{sorted(kwargs.items())}".encode()
        cache_key = hashlib.blake2b(cache_bytes, digest_size=16).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        result = func(*args, **kwargs)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        return result

    return wrapper

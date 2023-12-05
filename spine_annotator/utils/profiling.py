import time
from contextlib import contextmanager

profiling_enabled = True


def profile(fn):
    def wrapper(*args, **kwargs):
        if profiling_enabled:
            start_time = time.time()
        result = fn(*args, **kwargs)
        if profiling_enabled:
            elapsed_time = time.time() - start_time
            print(f"'{fn.__name__}': elapsed time={elapsed_time:.4f} seconds")
        return result

    return wrapper


@contextmanager
def profile_context(name):
    if profiling_enabled:
        start_time = time.time()
    yield
    if profiling_enabled:
        elapsed_time = time.time() - start_time
        print(f"{name}: elapsed time={elapsed_time:.4f} seconds")

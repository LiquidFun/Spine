import datetime
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_next_log_dir():
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir = log_dir / datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_dir.mkdir(exist_ok=True, parents=True)
    return log_dir


@lru_cache(maxsize=1)
def get_next_log_dir_on_tank():
    log_dir = Path("<log_dir>")
    log_dir = log_dir / datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_dir.mkdir(exist_ok=True, parents=True)
    return log_dir

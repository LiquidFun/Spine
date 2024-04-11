from functools import lru_cache

import numpy as np
from loguru import logger

from spine_segmentation.dataloader.statistics import Statistics
from spine_segmentation.resources.paths import MEASURE_STATISTICS_PATH, SECRETS_PATH, VECTOR_TABLE_PATH


@lru_cache
def get_measure_statistics() -> Statistics:
    logger.info(f"Loading measure statistics from {MEASURE_STATISTICS_PATH}")
    measure_statistics = Statistics(MEASURE_STATISTICS_PATH)
    logger.success(f"Loaded measure statistics: {measure_statistics}")
    return measure_statistics


@lru_cache
def get_vector_table() -> np.array:
    logger.info(f"Loading vector_table from {VECTOR_TABLE_PATH}")
    vector_table = np.load(VECTOR_TABLE_PATH)["vector_table"]
    logger.success(f"Loaded measure statistics: {vector_table.shape}")
    return vector_table


def get_secret(name):
    path = SECRETS_PATH / name
    if path.exists():
        return path.read_text().strip()

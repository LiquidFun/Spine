import numpy as np


class Spine:
    def __init__(self, data: np.ndarray, segmentation: np.ndarray):
        assert data.shape == segmentation.shape
        assert data.ndim == 3
        self.data = data
        self.segmentation = segmentation

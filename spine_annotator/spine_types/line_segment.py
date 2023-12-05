from typing import List, Union

import numpy as np


class LineSegment:
    def __init__(self, points: Union[np.ndarray, List[np.ndarray]], *, name=None):
        self.points = np.array(points)
        self.name = name

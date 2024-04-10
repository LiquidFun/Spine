import numpy as np


class Plane:
    def __init__(self, centroid: np.ndarray, normal_vector: np.ndarray, *, name: str = None, index: int = None):
        assert centroid.shape == (3,)
        assert normal_vector.shape == (3,)
        assert abs(np.linalg.norm(normal_vector)) > 1e-5
        self.centroid = centroid
        self.normal_vector = normal_vector / np.linalg.norm(normal_vector)

        if name is not None:
            self.name = name

        if index is not None:
            self.index = index

    @staticmethod
    def from_centroid_and_normal(centroid, normal_vector):
        return Plane(centroid, normal_vector)

    def as_corners(self, size=10) -> np.ndarray:
        v = self.normal_vector
        if np.linalg.norm(v - np.array([1, 0, 0])) > 1e-5:
            w = np.array([1, 0, 0])
        else:
            w = np.array([0, 1, 0])

        u = np.cross(v, w)
        u_prime = np.cross(v, u)

        vectors = np.array([u, u_prime, -u, -u_prime])
        return vectors * size + self.centroid

    @property
    def points(self):
        return self.as_corners()

    def are_points_above_plane(self, points):
        """
        >>> c = np.array([5, 5, 5])
        >>> n = np.array([1, 0, 0])
        >>> plane = Plane(c, n)
        >>> plane.are_points_above_plane(np.array([[0, 0, 1], [0, 0, 0], [0, 0, -1]]))
        array([ True, False, False])

        >>> import matplotlib.pyplot as plt
        >>> np.random.seed(0)
        >>> arr = np.random.randint(0, 2, size=(10, 10, 10)).astype(bool)
        >>> ax = plt.figure().add_subplot(projection="3d")
        >>> ticks = range(0, 12, 2)
        >>> ax.set_xticks(ticks)
        >>> ax.set_yticks(ticks)
        >>> ax.set_zticks(ticks)
        >>> is_above = plane.are_points_above_plane(np.array(np.nonzero(arr)).T)
        # >>> print(is_above)
        >>> colors = np.empty(arr.shape, dtype=object)
        >>> colors[np.nonzero(arr)] = ["#ff0000" if yes else "#0000ff" for yes in is_above]
        >>> ax.voxels(arr, facecolors=colors)
        {}
        >>> xx, yy = np.meshgrid(range(0, 11), range(0, 11))
        >>> zz = (n[0]*(xx - c[0]) + n[1]*(yy - c[1])) / (-n[2]) + c[2]
        >>> ax.plot_surface(xx, yy, zz, alpha=0.75, color="yellow")
        >>> plt.show(interactive=True)
        """
        assert points.shape[1] == 3
        return np.dot((points + 0.5) - self.centroid, self.normal_vector) >= 0

from typing import Dict, List, Tuple

import cc3d
import numpy as np

from spine_segmentation.dataloader.statistics import Statistics
from spine_segmentation.spine_types.labels import get_labels_for_n_classes
from spine_segmentation.spine_types.line_segment import LineSegment
from spine_segmentation.spine_types.plane import Plane


def mask_rare_rois(arr3d, min_volume: int) -> np.ndarray:
    arr3d = arr3d.copy()
    for i, count in zip(*np.unique(arr3d, return_counts=True)):
        if count < min_volume:
            arr3d[arr3d == i] = 0
    return arr3d


def separate_segmented_rois(
    arr3d: np.ndarray, min_volume=100
) -> np.ndarray:  # if any(name in str(path_out) for name in blacklist_num):
    # return False
    # arr3d[arr3d != 0] = 1

    # print(np.unique(arr3d,return_counts=True))
    seg, N = cc3d.connected_components(arr3d, connectivity=26, return_N=True)

    i_ = 1
    for i in range(1, N + 1):
        arr3d = seg == i_
        size = np.count_nonzero(arr3d)
        if size < min_volume:
            # skip this!
            # make to 0
            seg = np.where(arr3d, 0, seg)
            # shift >i down
            seg = np.where(seg > i_, seg - 1, seg)
            continue
        i_ += 1
    i_ -= 1
    # if i_ != 22:
    # print(color_error(f"found {i_} components from originally {N}."))
    # raise ValueError(f"Wrong number of components {i_}")
    # return False
    # order numbers from top to down
    N = i_

    def to_struct(i: int) -> Tuple:
        bool_map = seg == i
        pos = -np.average(np.where(bool_map)[2])
        num = np.sum(bool_map)
        return pos, num, bool_map, i

    ypos_map_idx = [to_struct(i) for i in range(1, N + 1)]
    if sorted(ypos_map_idx) != ypos_map_idx:
        # reorder indexes
        lst = sorted(ypos_map_idx)
        pos, num, bool_map, i = lst[0]
        merged_map = bool_map.astype(int)
        for idx, (pos_, num_, bool_map_, i_) in enumerate(lst[1:], start=2):
            merged_map = np.where(bool_map_, idx, merged_map)
        seg = merged_map
    return seg


def separate_segmented_rois_by_planes(arr3d: np.ndarray, planes: List[Plane], use_plane_index):
    seg = np.array(arr3d)
    coords = np.array(np.nonzero(seg)).T
    assert coords.shape[1] == 3

    # Set any point above, if there are any, as it does not have to be that all the point are below
    if use_plane_index:
        first_plane = planes[0]
        coords_above = coords[first_plane.are_points_above_plane(coords)].T
        seg[coords_above[0], coords_above[1], coords_above[2]] = first_plane.index - 1

    for plane in planes:
        coords_below = coords[~plane.are_points_above_plane(coords)].T
        # print(f"{coords_below.shape=}, {coords.shape=} {seg.shape=}")
        assert coords_below.shape[0] == 3
        if use_plane_index:
            seg[coords_below[0], coords_below[1], coords_below[2]] = plane.index + 1
        else:
            seg[coords_below[0], coords_below[1], coords_below[2]] += 1
    return seg


def __get_planes_from_discs_with_eigenvectors(separated_discs: np.ndarray, id_to_label=None):
    planes = []
    for i in np.unique(separated_discs)[1:]:
        coords = np.array(np.nonzero(separated_discs == i)).T
        centroid = np.mean(coords, axis=0)
        deviations = coords - centroid
        covariance_matrix = np.cov(deviations, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        normal_vector = eigenvectors[:, -1]
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        name = None if id_to_label is None else id_to_label[i]

        planes.append(Plane(centroid, normal_vector, name=name))
        for j in range(eigenvectors.shape[1]):
            points = [centroid, centroid + eigenvectors[:, j] * 10]
            planes.append(LineSegment(points, name=name + f"_eigen-{j + 1}"))
        # planes.append(LineSegment([centroid, centroid + normal_vector * 10], name=name + "normal-vec"))

    return planes


def _infer_additional_center_points_at_top_and_bottom(centroids: np.ndarray):
    centroids_as_numpy = np.array(centroids)[None, :, :]
    first4 = centroids_as_numpy[:, :4, :]
    last4 = centroids_as_numpy[:, -4:, :][
        :, ::-1, :
    ]  # Needs to be done in 2 steps, as otherwise -4::-1 has a different meaning
    first_predicted = Statistics.predict_next_center_point(first4)[:, -1, :]
    last_predicted = Statistics.predict_next_center_point(last4)[:, -1, :]

    return np.concatenate([first_predicted, centroids_as_numpy[0], last_predicted], axis=0)


def get_planes_from_rois(separated_rois: np.ndarray, id_to_label=None):
    planes = []
    centroids = []
    labels = []
    indices = []
    for i in np.unique(separated_rois)[1:]:
        coords = np.array(np.nonzero(separated_rois == i)).T
        centroid = np.mean(coords, axis=0)
        centroids.append(centroid)
        labels.append(str(i) if id_to_label is None else id_to_label[i])
        indices.append(i)

    if len(centroids) == 0:
        return []

    expanded = _infer_additional_center_points_at_top_and_bottom(centroids)

    # planes.append(LineSegment(expanded, name="spine_as_line"))

    for label, index, lower, current, upper in zip(labels, indices, expanded[2:], expanded[1:-1], expanded[:-2]):
        normal = upper - lower
        normal = normal / np.linalg.norm(normal)

        planes.append(Plane(current, normal, name=label, index=index))
        # planes.append(LineSegment([current, current+normal * 10], name=label + f"_normal"))

    return planes


def separate_rois_with_labels(arr3d: np.ndarray, *, split_along_discs=True) -> Tuple[np.ndarray, Dict[int, str]]:
    arr3d = arr3d.copy()
    assert arr3d.ndim == 3, f"Expected 3d array, got {arr3d.shape}"
    assert list(np.unique(arr3d)) == [
        0,
        1,
        2,
    ], f"Expected arr3d to be only 'background, vertebrae, discs', got: {np.unique(arr3d)}"

    groups = cc3d.connected_components(arr3d != 0, connectivity=26)
    uniq = np.unique(groups, return_counts=True)

    # Find the largest connected component in the image, this filters any small objects which are not connected
    # to the main spine object!
    # Take -2 because the most common will be the background
    most_common_id = sorted(zip(*uniq), key=lambda x: x[1])[-2][0]
    arr3d[groups != most_common_id] = 0

    if split_along_discs:
        discs = separate_segmented_rois(arr3d == 2)
        planes = get_planes_from_rois(discs)
        # vertebrae = separate_segmented_rois(arr3d == 1)
        vertebrae = separate_segmented_rois_by_planes((arr3d == 1).astype(int), planes, use_plane_index=False)
    else:
        discs = separate_segmented_rois(arr3d == 2)
        vertebrae = separate_segmented_rois(arr3d == 1)

    # assert len(np.unique(vertebrae)) == len(VERTEBRA_LABELS) + 1, f"Expected {len(VERTEBRA_LABELS) + 1} vertebrae, got {len(np.unique(vertebrae))}"
    # assert len(np.unique(discs)) == len(DISC_LABELS) + 1, f"Expected {len(DISC_LABELS) + 1} discs, got {len(np.unique(discs))}"

    instances = (vertebrae * 2) - (vertebrae != 0) + (discs * 2)
    # print(list(zip(*np.unique(instances, return_counts=True))))

    ids = np.unique(instances)[1:]
    labels = get_labels_for_n_classes(49)
    id_to_label = dict(zip(range(1, 50), labels))
    id_to_label = {id: label for id, label in id_to_label.items() if id in ids}
    return instances, id_to_label

    from matplotlib import pyplot as plt

    plt.imshow(discs[8])
    plt.show()

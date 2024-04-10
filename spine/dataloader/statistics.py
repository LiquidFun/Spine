import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class Statistics:
    def __init__(self, measure_statistics_path):
        self.raw_statistics = pd.read_csv(measure_statistics_path)
        # self.all_namings, self.all_namings_no_duplicates, self.roi_naming_lookup = self._get_sorted_namings()

        self.nan_filtered_statistics = self._filter_empty_columns(self.raw_statistics)
        self.roi_statistics, self.dicom_metadata, self.spinal_canal_statistics = self._split_statistics(
            self.nan_filtered_statistics
        )
        self.statistics = self._get_statistics_without_rare_rois(self.roi_statistics)

        self.stats_per_roi = self.filter_by_column(
            self.statistics, excluded_substrings=["ANGLE_COR", "ANGLE_SAG", "HEIGHT_MIN"], reshape=True
        )

        self.centers = self.filter_by_column(self.statistics, ["CENTER"], reshape=True)

        # Calculate vectors

        self.disc_direction_vectors = self.filter_by_column(
            self.statistics, ["DIRECTION"], ["CANAL_DIRECTION"], reshape=True
        )
        self.canal_direction_vectors = self.filter_by_column(self.statistics, ["CANAL_DIRECTION"], reshape=True)
        # self.canal_direction_vectors = self.filter_by_column(
        #     self.statistics, ["VERTEBRA_CANAL_DIRECTION", "DISC_DIRECTION"], reshape=True
        # )
        self.roi_diff_vectors = self._calculate_roi_diff_vectors(self.centers)

        self.shape_printable = "\n".join(
            [
                f"\t{self.raw_statistics.shape=}",
                f"\t{self.statistics.shape=}",
                f"\t{self.centers.shape=}",
                f"\t{self.disc_direction_vectors.shape=}",
                f"\t{self.canal_direction_vectors.shape=}",
                f"\t{self.roi_diff_vectors.shape=}",
            ]
        )
        # s = Counter([c.split(";")[1] for c in full_statistics.columns])
        # print(s)

        # patient_centers = pca_normalise(patient_centers)

    # ===================================================
    # Public
    # ===================================================

    @property
    def all_vectors(self):
        return [self.disc_direction_vectors, self.canal_direction_vectors, self.roi_diff_vectors]

    @property
    def roi_naming_lookup(self):
        # vertebrae = [f"{c}{i+1}" for c, count in zip("CTLS", [7, 12, 5, 1] for i in range(count))][1:]
        # discs = [f"{c1}-{c2[1:] if c1[0] == c2[0] else c2}" for ]
        c = "C2 C2-3 C3 C3-4 C4 C4-5 C5 C5-6 C6 C6-7 C7 C7-T1 "
        t = "T1 T1-2 T2 T2-3 T3 T3-4 T4 T4-5 T5 T5-6 T6 T6-7 T7 T7-8 T8 T8-9 T9 T9-10 T10 T10-11 T11 T11-12 T12 T12-L1 "
        l = "L1 L1-2 L2 L2-3 L3 L3-4 L4 L4-5 L5 L5-S1 S1 "
        return dict(zip((c + t + l).split(), range(1000)))

    def filter_by_column(self, statistics=None, required_substrings=None, excluded_substrings=None, reshape=False):
        """Return shape (10k, 51 * #matching cols), if reshape==True, return shape (10k, 51, #matching cols)"""

        if statistics is None:
            statistics = self.statistics
        if required_substrings is None:
            required_substrings = statistics.columns
        if excluded_substrings is None:
            excluded_substrings = []

        relevant_columns = [
            column
            for column in statistics.columns
            if any(c in column for c in required_substrings) and all(c not in column for c in excluded_substrings)
        ]

        # namings = VectorAnnotator().roi_naming_lookup
        # sorted_columns = sorted(relevant_columns, key=lambda c: namings[c.split(";")[0]])

        combined_roi_xyz = statistics[relevant_columns]
        if reshape:
            separate_roi_xyz = self._reshape_combined_roi_xyz_to_separate_coordinates(combined_roi_xyz)
            return separate_roi_xyz
        return combined_roi_xyz

    def __str__(self):
        return "\nStatistics:\n" + self.shape_printable

    # ===================================================
    # Private
    # ===================================================

    def _split_statistics(self, statistics):
        """Split statistics into a list of statistics for each patient"""
        dicom_metadata = self.filter_by_column(statistics, ["DICOM;"])
        spinal_canal_statistics = self.filter_by_column(statistics, ["spinal_canal;"])

        used_columns = dicom_metadata.columns.tolist() + spinal_canal_statistics.columns.tolist()
        rest = self.filter_by_column(statistics, None, used_columns)
        return rest, dicom_metadata, spinal_canal_statistics

    def _filter_empty_columns(self, statistics):
        """Filter columns that are all NaN"""
        statistics = self.filter_by_column(statistics, None, ["unexpected_"])
        nan_filtered = statistics.dropna(axis=1, how="all")
        # nan_filtered = nan_filtered.dropna(axis=0, how="all")
        # nan_filtered = statistics.fillna(0.0)
        # assert nan_filtered.isna().sum().sum() == 0
        return nan_filtered

    def _get_statistics_without_rare_rois(self, measure_statistics):
        must_not_be_in = ["unexpected", "L1-S1", "L2-S1", "L3-S1", "spinal_canal"]
        statistics = self.filter_by_column(measure_statistics, None, must_not_be_in)

        centers = self.filter_by_column(statistics, ["CENTER"])

        skip_nan_percentage = 0.6
        column_nan_percentage = ((np.isnan(centers).sum(axis=0)) / centers.shape[0]) > skip_nan_percentage
        skip_rois = set(column.split(";")[0] for column in column_nan_percentage[column_nan_percentage].axes[0])
        remaining_columns = [column for column in statistics.columns if column.split(";")[0] not in skip_rois]

        sorted_columns = sorted(remaining_columns, key=lambda c: self.roi_naming_lookup.get(c.split(";")[0], 1e9))

        return statistics[sorted_columns]

    def _reshape_combined_roi_xyz_to_separate_coordinates(self, combined_roi_xyz):
        """Input shape: (10k, 153), output shape: (10k, 51, 3)"""

        remove_prefixes = "|".join(["DISC_", "VERTEBRA_"])
        # remove_prefixes = "|".join(["DISC_", "VERTEBRA_", "CANAL_"])

        counts = Counter([re.search(rf";({remove_prefixes})*(.*)", c).group(2) for c in combined_roi_xyz.columns])
        assert len(set(counts.values())) == 1, f"Column-sizes are not the same: {counts}"
        s = combined_roi_xyz.shape[1] // counts.most_common(1)[0][1]

        separate_roi_xyz = np.zeros((combined_roi_xyz.shape[0], combined_roi_xyz.shape[1] // s, s))
        for i in range(s):
            separate_roi_xyz[:, :, i] = combined_roi_xyz.iloc[:, i : combined_roi_xyz.shape[1] : s]
        return separate_roi_xyz

    def _pca_normalise(self, patient_centers):
        """Deprecated: use _sphere_coordinates_normalise instead"""

        pca = PCA(n_components=3)
        # pca = TruncatedSVD(n_components=3)
        # median_size = centers.mean(axis=1)
        for i in range(len(patient_centers)):
            if i % 1000 == 0:
                print(i)
            # current_centers = patient_centers[i]
            non_nan_centers = patient_centers[i][~np.isnan(patient_centers[i]).any(axis=1)]
            fitted = pca.fit(non_nan_centers)
            components = fitted.components_.T
            for j in range(3):
                if np.dot(components, np.array([1, 1, 1]))[j] < 0:
                    components[j] *= -1
            # patient_centers[i] -= fitted.mean_
            patient_centers[i] = patient_centers[i] @ components
            # current_centers -= fitted.mean_
            # print(fitted.components_)
            # current_centers /= fitted.components_[0]
        return patient_centers

    def _sphere_coordinates_normalise(self, patient_centers):
        # Shape: (10k, 51, 3)
        patient_centers -= patient_centers[:, -1][:, np.newaxis, :]

        # Flip axis because there were inconsistencies when some points were on the other side of the y axis
        x, y, z = 2, 0, 1

        # return patient_centers
        spherical = np.zeros(patient_centers.shape)
        spherical[:, :, x] = np.linalg.norm(patient_centers, axis=2)
        spherical[:, :, y] = np.arccos(patient_centers[:, :, z] / spherical[:, :, x])
        spherical[:, :, z] = np.arctan2(patient_centers[:, :, y], patient_centers[:, :, x])

        center_nth_roi = 0
        r = spherical[:, center_nth_roi, x]
        theta = spherical[:, center_nth_roi, y]
        phi = spherical[:, center_nth_roi, z]

        spherical[:, :, x] /= np.nan_to_num(r)[:, np.newaxis]
        # spherical[:, :, 0] = 10

        spherical[:, :, y] -= theta[:, np.newaxis] - np.pi / 2
        spherical[:, :, z] -= phi[:, np.newaxis]
        # spherical[:, :, 1] += (spherical[:, :, 2] > np.pi) * np.pi
        # spherical[:, :, 2] -= (spherical[:, :, 2] > np.pi) * np.pi

        # spherical[:, :, y] += .1

        new_centers = np.zeros(patient_centers.shape)
        new_centers[:, :, x] = spherical[:, :, x] * np.sin(spherical[:, :, y]) * np.cos(spherical[:, :, z])
        new_centers[:, :, y] = spherical[:, :, x] * np.sin(spherical[:, :, y]) * np.sin(spherical[:, :, z])
        new_centers[:, :, z] = spherical[:, :, x] * np.cos(spherical[:, :, y])

        return new_centers

    @staticmethod
    def predict_next_center_point(centers):
        """Input shape: (10k, 51, 3), output shape: (10k, 51+1, 3)"""

        x, y, z = centers[:, :, 0], centers[:, :, 1], centers[:, :, 2]

        # x_axis = np.arange(x.shape[1])[np.newaxis].repeat(x.shape[0], axis=0)
        x_axis = np.arange(x.shape[1])

        def fit(input_axis):
            optimal_params = np.polyfit(x_axis, input_axis.T, 2)
            poly = np.polyval(optimal_params, -1)
            return poly

        next_x, next_y, next_z = fit(x), fit(y), fit(z)
        new_centers = np.stack([next_x, next_y, next_z], axis=1)[:, None, :]
        return np.concatenate([centers, new_centers], axis=1)

    def _calculate_roi_diff_vectors(self, centers):
        centers[np.isnan(centers)] = np.mean(centers[~np.isnan(centers)])

        centers_with_new_top = self.predict_next_center_point(centers)
        vectors = centers_with_new_top[:, :-1] - centers_with_new_top[:, 1:]

        # normalise
        vectors = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
        return vectors

    def _get_sorted_namings(self, discard_below_percentage=0.01) -> Tuple[List[str], List[str], Dict[str, int]]:
        # All namings sorted
        all_namings = set()
        counts = {}
        for column in self.raw_statistics.columns:
            if any(part in column for part in ["CENTER"]):
                roi_name = column.split(";")[0]
                all_namings.add(roi_name)
                counts[roi_name] = self.raw_statistics[column].count()
        all_namings -= {"spinal_canal", "unexpected_0"}
        all_namings = [n for n in all_namings if counts[n] / len(self.raw_statistics) > discard_below_percentage]
        all_namings += {"S1"}
        [(n, counts[n] / len(self.raw_statistics)) for n in all_namings]
        all_namings_sorted = []
        for current_part_of_spine in "CTLS":
            sort_by = lambda name: int(name[1:].split("-")[0]) + 0.5 * ("-" in name)
            all_namings_sorted.extend([n for n in sorted(all_namings, key=sort_by) if current_part_of_spine == n[0]])

        # Roi naming lookup
        roi_naming_lookup: Dict[str, int] = {}
        vertebra_index = 0
        for naming in all_namings_sorted:
            roi_naming_lookup[naming] = vertebra_index - ("-" in naming)
            vertebra_index += 2 * ("-" not in naming)

        # All namings no duplicates
        naming_occ = Counter(roi_naming_lookup.values())
        to_be_removed = [n for n in all_namings if naming_occ[roi_naming_lookup.get(n, 0)] >= 2 and "-S1" in n]
        all_namings_no_duplicates = list(filter(lambda n: n not in to_be_removed, all_namings_sorted))
        return all_namings_sorted, all_namings_no_duplicates, roi_naming_lookup

import concurrent.futures.process
import datetime
import random
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from tqdm import tqdm

from spine_segmentation.resources.paths import PLOTS_PATH, RESULTS_PATH
from spine_segmentation.resources.preloaded import get_measure_statistics
from spine_segmentation.spine_types.spine import Spine


def _file_with_fallback(file: Path) -> Path:
    if file.exists():
        return file
    logger.info("cannot find file, fallback: check in resource path")
    file_fallback = Path(__file__).parent.joinpath(file).resolve()
    if file_fallback.exists():
        logger.info(f"found file in resource path {file_fallback}.")
        return file_fallback
    logger.info("did not find file in resource path.")
    return file


@dataclass
class AnnotatorBase(ABC):
    debug_dir: Path = None

    @abstractmethod
    def _process(self, spine: Spine):
        raise NotImplementedError()

    def process(self, spine: Spine):
        """ """
        # self._algorithm_validation()
        res = self._process(spine)
        return res


class AnnotatorSimple(AnnotatorBase, ABC):
    """Annotates spine rois by labeling their indices"""

    stats = None
    statistics = None
    available_properties = ["DIRECTION"]  # , "CENTER", "VOLUME"]
    properties_to_be_used = available_properties

    # ================================================================================
    #  Initialization
    # ================================================================================

    def __init__(self, seed=None, load=True, save_to=None):
        logger.info("Initializing Annotator; this may take a while")
        if self.__class__.statistics is None:
            self.__class__.stats = self._load_measure_statistics()
            self.__class__.statistics = self.stats.statistics
        self.variable_lookup = dict(zip(self.available_properties, range(100)))
        self.property_indices = [self.variable_lookup[key] for key in self.properties_to_be_used]
        self.all_namings = list(self.stats.roi_naming_lookup.keys())
        self.all_namings_no_duplicates = list(self.stats.roi_naming_lookup.keys())
        self.roi_naming_lookup = self.stats.roi_naming_lookup
        # self.vector_table = self.stats.filter_by_column(
        # required_substrings=["DIRECTION_"], excluded_substrings=["CANAL_"], reshape=True
        # )[:, :, None, :]

        # self.vector_table = self.stats.roi_diff_vectors[:, :, None, :]
        self.vector_table = self.stats.canal_direction_vectors[:, :, None, :]

        # if load:
        #     self.vector_table = get_vector_table()
        # else:
        #     self.vector_table = self._normalize(self._initialize_vector_lookup_table())
        if save_to is not None:
            np.savez_compressed(save_to, vector_table=self.vector_table)
            print(f"Saving pre-computed vector table to {save_to}")
        self.max_roi_index = max(self.roi_naming_lookup.values())
        if seed is not None:
            np.random.seed(seed)
        self._validate_init()

    def _validate_init(self) -> None:
        """Validates properties which are assumed in the rest of the class"""
        assert all(("-" in naming) == i % 2 for naming, i in self.roi_naming_lookup.items()), "Not all discs are odd"

    def _get_sorted_namings(self, discard_below_percentage=0.01) -> Tuple[List[str], List[str], Dict[str, int]]:
        # All namings sorted
        all_namings = set()
        counts = {}
        for column in self.statistics.columns:
            if any(part in column for part in self.variable_lookup.keys()):
                roi_name = column.split(";")[0]
                all_namings.add(roi_name)
                counts[roi_name] = self.statistics[column].count()
        all_namings -= {"spinal_canal", "unexpected_0"}
        all_namings = [n for n in all_namings if counts[n] / len(self.statistics) > discard_below_percentage]
        all_namings += {"S1"}
        [(n, counts[n] / len(self.statistics)) for n in all_namings]
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

    def _load_measure_statistics(self) -> pd.DataFrame:
        return get_measure_statistics()

    def _initialize_vector_lookup_table(self) -> np.array:
        relevant_columns = []
        for column in self.statistics.columns:
            # if 'DIRECTION' in column or "POSITION" in column:
            if any(part in column for part in self.variable_lookup.keys()):
                if all(part not in column for part in ["DIFF", "PROB"]):
                    relevant_columns.append(column)
        index = -1
        coordinate_index_lookup = {"X": 0, "Y": 1, "Z": 2, "ALL": (0, 1, 2)}
        shape_4d_array = (
            len(self.statistics),
            max(self.roi_naming_lookup.values()) + 1,
            len(self.variable_lookup),
            3,
        )
        patient_vectors = np.zeros(shape_4d_array)
        for row_index, row in list(self.statistics[relevant_columns].iterrows()):
            if row_index % 1000 == 0:
                logger.info(f"Annotator at row: {row_index}/{len(self.statistics)}")
            for column, value in row.items():
                naming, variable = column.split(";")
                if variable.endswith("VOLUME"):
                    variable += "_ALL"
                if not pd.isna(value) and naming in self.roi_naming_lookup:
                    multi_index = (
                        row_index,
                        self.roi_naming_lookup[naming],
                        self.variable_lookup[variable.split("_")[-2]],
                        coordinate_index_lookup[variable.split("_")[-1]],
                    )
                    assert np.all(patient_vectors[multi_index] == 0), "Array is not zero!"
                    patient_vectors[multi_index] = value

        # Filter some rows which have too many zero rows:
        max_allowed_all_0_entries = 4
        indices_to_be_filtered = (patient_vectors == 0).all(axis=(2, 3)).sum(axis=1) > max_allowed_all_0_entries
        patient_vectors = np.delete(patient_vectors, indices_to_be_filtered, axis=0)

        return patient_vectors

    def _normalize(self, vector_table):
        """Input either all patients shape (8512, 52, 2, 3) or a single patient shape (52, 2, 3)"""

        def divide_by_max(variable: str):
            max_val = vector_table[..., self.variable_lookup[variable], :].max(axis=(1, 2))
            vector_table[..., self.variable_lookup[variable], :] /= max_val.reshape(-1, 1, 1)

        def divide_by_global_max(variable: str):
            max_val = vector_table[..., self.variable_lookup[variable], :].max(axis=(0, 1, 2))
            vector_table[..., self.variable_lookup[variable], :] /= max_val

        def subtract_min(variable: str):
            min_val = vector_table[..., self.variable_lookup[variable], :].min(axis=(1, 2))
            vector_table[..., self.variable_lookup[variable], :] -= min_val.reshape(-1, 1, 1)

        is_single_vector = len(vector_table.shape) == 3
        if is_single_vector:
            vector_table = vector_table[np.newaxis, ...]

        if "VOLUME" in self.available_properties:
            divide_by_max("VOLUME")
        if "CENTER" in self.available_properties:
            subtract_min("CENTER")
            divide_by_max("CENTER")

        if is_single_vector:
            vector_table = vector_table[0]

        return vector_table

    # ================================================================================
    #  Samples & validation
    # ================================================================================

    def _evaluate_sample(self, size: int, patient_index=None, start_at=None):
        if patient_index is None:
            patient_index = random.randint(0, len(self.statistics) - 1)
        sample, start_at = self._get_random_sample(patient_index, size, start_at)
        table_without_sample = self._get_table_without_sample(patient_index)
        match = self._find_best_match(table_without_sample, sample, is_disc=start_at % 2 == 1, correct=start_at)
        # print(f"{match=} {start_at=} {size=}")
        return self.roi_naming_lookup[match[0]], start_at
        # if self.roi_naming_lookup[match[0]] == start_at:
        #     correct[size] += 1
        # correct_roi[start_at] = (correct_roi[start_at][0] + (match == start_at), correct_roi[start_at][1] + 1)

    def evaluate_each_roi_on_patient(self, size: int, patient_index: int = None):
        off_by = defaultdict(int)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            to = self.max_roi_index - size
            args = [[size] * to, [patient_index] * to, range(0, to)]
            for guess, actual in tqdm(executor.map(self._evaluate_sample, *args)):
                off_by[actual] = guess
        return off_by

    def _algorithm_validation(self, iterations=100):
        """Tests random samples from NAKO on the entire NAKO dataset

        Each test is validated without the current sample. The start position is taken at random (can be disc or
        vertebra), each size from 1 to about 30 is tried with #iterations (e.g. 100).
        """
        correct = defaultdict(int)
        correct_plus_minus_1 = defaultdict(int)
        correct_roi = defaultdict(lambda: (0, 0))

        date = datetime.date.today().strftime("%Y-%m-%d")
        model_description = "canal_direction"
        postfix = f"{date}_{model_description}"

        with concurrent.futures.ProcessPoolExecutor(max_workers=160) as executor:
            for size in range(1, self.max_roi_index):
                for guess, actual in tqdm(executor.map(self._evaluate_sample, [size] * iterations)):
                    correct[size] += guess == actual
                    correct_plus_minus_1[size] += abs(guess - actual) <= 2
                    correct_roi[actual] = (correct_roi[actual][0] + (guess == actual), correct_roi[actual][1] + 1)
                print(f"correct: {dict(correct)}/{iterations}")
                print(f"corr+-2: {dict(correct_plus_minus_1)}/{iterations}")
                table = pd.DataFrame(
                    {
                        "slice size": range(1, size + 1),
                        "correct (%)": correct.values(),
                        "correct ±2 (%)": correct_plus_minus_1.values(),
                    }
                )
                print(table.to_markdown(index=False))

        table.to_csv(RESULTS_PATH / f"roi_accuracy_{postfix}.csv")
        self._plot_roi_accuracy(correct_roi, postfix)
        self._plot_slice_accuracy(correct, correct_plus_minus_1, iterations, postfix)

    def _plot_slice_accuracy(self, correct: Dict, correct_plus_minus_1: Dict, iterations: int, postfix: str):
        plt.bar(
            correct_plus_minus_1.keys(),
            [p / iterations for p in correct_plus_minus_1.values()],
            label="accuracy ±2",
            color="orange",
        )
        plt.bar(correct.keys(), [p / iterations for p in correct.values()], label="accuracy", color="green")
        plt.legend()
        plt.title(f"Accuracy of Vector Annotator by slice size ({postfix})")
        plt.ylabel("Accuracy %")
        plt.xlabel("Slice size")
        plt.xticks(list(correct.keys()))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.savefig(PLOTS_PATH / f"slice_accuracy_{postfix}.png")
        logger.info("Annotator validation complete")

    def _plot_roi_accuracy(self, correct_roi: Dict, postfix):
        keys, values = list(zip(*sorted(correct_roi.items())))
        values = [correct / total for correct, total in values]
        plt.bar(keys, values)
        plt.xticks(keys)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.ylabel("% of correct ROIs")
        plt.xlabel("ROI index")
        plt.title(f"Accuracy of Vector Annotator by ROI index ({postfix})")
        plt.savefig(PLOTS_PATH / f"roi_accuracy_{postfix}.png")
        plt.show()

    def _get_random_sample(self, patient_index: int, size=None, start_at=None):
        if size is None:
            size = random.randint(3, self.max_roi_index)
        if start_at is None:
            start_at = random.randint(0, self.max_roi_index - size)
        vector = self.vector_table[patient_index, start_at : start_at + size][..., self.property_indices, :]
        return vector, start_at

    def _get_subsets_at(self, table, start_at, size) -> List[Tuple[np.ndarray, List[str]]]:
        """Returns a list of tuples of subsets and namings

        * In total there are 1 or 2 subsets. 2 are returned only when it is ambiguous whether the last element is S1
        * The namings are the corresponding naming associated with this subset
        """
        subset_no_s1 = table[:, start_at : start_at + size, self.property_indices, :]
        naming_no_s1 = self.all_namings_no_duplicates[start_at : start_at + size]
        subsets = [(subset_no_s1, naming_no_s1)]
        s1_index = self.roi_naming_lookup.get("S1", None)
        if s1_index is not None:
            for name, index in self.roi_naming_lookup.items():
                # Check for discs which lead to S1
                is_second_to_last_index = index + 1 == start_at + size - 1
                if "-S1" in name and is_second_to_last_index and s1_index != index + 1:
                    indices_with_s1 = list(range(start_at, start_at + size - 1)) + [s1_index]
                    naming_with_s1 = naming_no_s1[:-1] + ["S1"]
                    if len(naming_with_s1) >= 2:
                        naming_with_s1[-2] = "-".join(naming_with_s1[-2].split("-")[:1] + ["S1"])
                    # Workaround with ..., because [:, indices_with_s1, self.property_indices, :] produces
                    # shape (x, 3, 2) due to advanced indexing (when 2 lists or tuples are used then an 'and'
                    # conjunction is used between them
                    subsets.append((table[:, indices_with_s1][..., self.property_indices, :], naming_with_s1))

        for i, (subset, naming) in enumerate(subsets):
            indices_to_be_removed = (subset == 0).all(axis=(2, 3)).any(axis=1)
            subsets[i] = (np.delete(subset, indices_to_be_removed, axis=0), naming)
        # assert all(subset.shape == subsets[0][0].shape for subset, _ in subsets), "Subsets are not of the same shape!"
        assert all(all(name in self.all_namings for name in naming) for _, naming in subsets), "Unknown name!"
        return subsets

    def _get_table_without_sample(self, filter_index: int):
        table_without_sample = self.vector_table[[i for i in range(len(self.vector_table)) if i != filter_index]]
        assert len(table_without_sample) + 1 == len(self.vector_table)
        return table_without_sample

    # ================================================================================
    #  Matching
    # ================================================================================

    @abstractmethod
    def _find_best_match(self, table: np.ndarray, sample: np.ndarray, is_disc: bool, correct: int) -> List[str]:
        ...

    def _is_naming_incomplete(self, spine) -> bool:
        return any(roi.name == "" for roi in (spine.vertebras + spine.discs))

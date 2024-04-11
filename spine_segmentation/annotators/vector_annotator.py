import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from spine_segmentation.annotators.annotator_base import AnnotatorSimple
from spine_segmentation.spine_types.disc import Disc
from spine_segmentation.spine_types.roi import Roi
from spine_segmentation.spine_types.spine import Spine

logger = logging.getLogger(__name__)


@dataclass
class _SubsetErrorScore:
    score: float
    index: int
    subset: np.ndarray
    naming: List[str]

    def __lt__(self, other):
        return self.score < other.score


# _SubsetErrorScore = namedtuple("_SubsetErrorScore", ["score", "index", "subset", "naming"])
def _plot_error_for_each_start_index(error_scores: List[_SubsetErrorScore], size: int, correct: int, debug_dir: Path):
    best, *others = sorted(error_scores)
    second_best = _SubsetErrorScore(best.score / 10, -1, None, None) if len(others) == 0 else others[0]
    index_to_score = defaultdict(lambda: 1e100)
    for error_score in error_scores:
        index_to_score[error_score.index] = min(index_to_score[error_score.index], error_score.score)
    starts = sorted(index_to_score)
    scores = list(zip(*sorted(index_to_score.items())))[1]
    certainty = 1 - best.score / second_best.score
    color_lookup = defaultdict(lambda: ("b", "Any"))
    color_lookup.update(
        {
            best.index: ("r", "Best guess (incorrect)"),
            second_best.index: ("y", "Second best guess"),
            correct: ("g", "Actual"),
        }
    )
    colors = [color_lookup[k][0] for k in starts]
    plt.bar(starts, scores, color=colors)
    plt.xticks(starts)
    color_and_label = set(color_lookup.values())
    plt.legend(handles=[matplotlib.patches.Patch(color=col, label=label) for col, label in color_and_label])
    plt.xlabel("ROI start index")
    plt.ylabel("Error (smallest is best)")
    plt.title(f"Error for sample of {size=}, {certainty=:.1%} {'CORRECT' if correct == best.index else 'WRONG'}")
    plt.savefig(debug_dir / "annotator.png")
    plt.cla()


class VectorAnnotator(AnnotatorSimple):
    def _find_best_match(self, table, sample, is_disc: bool = None, correct=-1, debug_dir: Path = None) -> List[str]:
        size = len(sample)
        error_scores: List[_SubsetErrorScore] = []
        earliest_possible_start = 0 if is_disc is None else int(is_disc)
        latest_possible_start = table.shape[1] - size + 1
        scoring_function = self._get_scoring_function()
        for start in range(earliest_possible_start, latest_possible_start, 1 if is_disc is None else 2):
            for subset, naming in self._get_subsets_at(table, start, size):
                error_scores.append(_SubsetErrorScore(scoring_function(subset, sample), start, subset, naming))
        if debug_dir is not None:
            _plot_error_for_each_start_index(error_scores, size, correct, self.debug_dir)
        best_error_score = sorted(error_scores)[0]
        return best_error_score.naming

    def _plot_vectors_for_best_guess(self, table, error_scores) -> List:
        pass

    @staticmethod
    def _mse_sample_on_subset(sample, subset):
        if len(sample.shape) != 3:
            raise Exception(f"{len(sample.shape)=} != 3")
        if len(subset.shape) != 4:
            raise Exception(f"{len(subset.shape)=} != 4")
        diff = subset - sample  # e.g. shape: (8511, 21, 2, 3)
        return np.average(diff**2, axis=tuple(range(1, len(diff.shape))))  # e.g. shape: (8511)

    @staticmethod
    def _get_scoring_function(strategy="mse_top_n", n=50) -> Callable[[np.ndarray, np.ndarray], float]:
        """Returns a function which takes a subset and a sample and returns some error value

        Subset is of shape [#nako_patients, #rois, #properties, 3], e.g. [8511, 21, 2, 3]
        Sample is of shape [#rois, #properties, 3], e.g. [21, 2, 3]

        """

        mse_scaling = (np.cos(np.pi * np.arange(n) / (n - 1)) + 1) / 2.5 + 0.2

        def mse_top_n(subset, sample) -> float:
            mse = VectorAnnotator._mse_sample_on_subset(sample, subset)  # e.g. shape: (8511)
            min_values = np.sort(np.partition(mse, n)[:n]) * mse_scaling  # e.g. shape: (50)
            return np.average(min_values)  # e.g. shape: (1)

        def naive_all_sum(subset, sample) -> float:
            diff = subset - sample
            return np.log(np.sum(diff**2))

        return {
            "mse_top_n": mse_top_n,
            "naive_all_sum": naive_all_sum,
        }[strategy]

    def _ensure_alternating_rois(self, rois: List[Roi]):
        """Ensures that the rois are alternating between disc and vertebra

        For this it creates a new roi if there are two consecutive rois of the same type.
        The other rois are unchanged, and their index would be incorrect.
        Sometimes there is a missing roi, which actually exists but was to segmented correctly.
        The missing roi is interpolated from the two surrounding rois of the same type (if this is not possible then
        any adjacent 2 rois are used)
        """

        @dataclass
        class FakeRoi:
            direction: List[float] = None
            volume: float = None

        i = 1
        while i < len(rois):
            # Check if the rois are of the same class
            above, below = rois[i - 1], rois[i]
            above2, below2 = above, below
            if 0 <= i - 2 <= i + 1 < len(rois):
                above2, below2 = rois[i - 2], rois[i + 1]
            if above.__class__ == below.__class__:
                # If so, create a new FakeRoi and insert it there
                direction = (above.direction + below.direction) / 2
                direction = direction / np.linalg.norm(direction)
                volume = (above2.volume + below2.volume) / 2
                rois.insert(i, FakeRoi(direction, volume))
            i += 1

    def _make_vector_from_spine(self, spine) -> Tuple[np.ndarray, bool]:
        """Returns shape (#rois, #properties, 3) and whether the first roi is a disc"""
        rois = spine.get_rois(sort=True, skip_canal=True).copy()
        self._ensure_alternating_rois(rois)
        rois_as_vectors = np.zeros((len(rois), len(self.properties_to_be_used), 3))
        for roi_index, roi in enumerate(rois):
            for index, measure_name in enumerate(self.properties_to_be_used):
                rois_as_vectors[roi_index, index, :] = getattr(roi, measure_name.lower())
        is_disc = isinstance(rois[0], Disc)
        return rois_as_vectors, is_disc

    # def _apply_new_spine_roi_naming(self, spine: Spine, naming: List[str]):
    #     rois = spine.get_rois(sort=True, skip_canal=True)
    #     if len(rois) != len(naming):
    #         logger.warning(f"There is likely a roi missing in the spine {spine.uid}, interpolating it for naming!")
    #     roi_index = 0
    #     for naming_index in range(len(naming)):
    #         roi = rois[roi_index]
    #         if isinstance(roi, Disc) and "-" in naming[naming_index]:
    #             roi.name = naming[naming_index]
    #             roi_index += 1
    #         elif isinstance(roi, Vertebra) and "-" not in naming[naming_index]:
    #             roi.name = naming[naming_index]
    #             roi_index += 1
    #     # for roi, name in zip(rois, naming):
    #     #    roi.name = name
    #     logger.info(f"sort rois: {[r.name for r in rois]}")

    def _process(self, spine: Spine):
        if self._is_naming_incomplete(spine):
            logger.info(f"Naming is incomplete for patient {spine.uid}")
            spine_vector, is_disc = self._make_vector_from_spine(spine)
            spine_vector = self._normalize(spine_vector)
            naming_proposal = self._find_best_match(self.vector_table, spine_vector, is_disc=is_disc)
            if naming_proposal[-1] in ("L6", "L5", "L4"):
                naming_proposal[-1] = "S1"
                naming_proposal[-2] = f"{naming_proposal[-3]}-S1"
            self._apply_new_spine_roi_naming(spine, naming_proposal)


if __name__ == "__main__":
    annotator = VectorAnnotator()
    annotator._algorithm_validation(iterations=1000)

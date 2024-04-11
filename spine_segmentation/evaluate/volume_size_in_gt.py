import random

from spine_segmentation.cli.cli import colored_tracebacks
from spine_segmentation.instance_separation.instance_separation import separate_rois_with_labels
from spine_segmentation.resources.paths import TRAIN_SPLIT_CSV_PATH, VAL_SPLIT_CSV_PATH

colored_tracebacks()

from collections import defaultdict

import numpy as np

from spine_segmentation.datasets.segmentation_dataset import SegmentationDataModule


def volume_size():
    val_dataloader = SegmentationDataModule(
        [TRAIN_SPLIT_CSV_PATH, VAL_SPLIT_CSV_PATH],
        slice_wise=False,
        batch_size=1,
        # crop_height_to_px=416,
        target_shape=(18, 320, 896),
    ).train_dataloader()

    random.seed(0)
    nums = defaultdict(list)
    mins = defaultdict(lambda: 1e9)
    sums = defaultdict(lambda: 0)
    avg = defaultdict(lambda: 0)
    for index, (image, gt) in enumerate(val_dataloader):
        print(f"\n\n==================== Processing sample {index} =====================\n")
        print(image.shape)
        instances, labels = separate_rois_with_labels(gt.detach().cpu().numpy()[0], split_along_discs=False)
        for i, count in list(zip(*np.unique(instances, return_counts=True)))[1:]:
            l = labels.get(i, i)
            nums[l].append(count)
            mins[l] = min(mins[l], count)
            sums[l] += count
            avg[l] = sums[l] / (index + 1)
            print(
                f"{labels.get(i, 'NOO'):<8} {count:<7} {i:5=} min={mins[l]:<9} avg={avg[l]:<9.0f} sum={sums[l]:<9} {sorted(nums[l])[:5]}"
            )


def main():
    volume_size()


if __name__ == "__main__":
    main()

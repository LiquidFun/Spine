from pathlib import Path

from spine_segmentation import SegmentationInference
from spine_segmentation.datasets.sample import SampleIterator
from spine_segmentation.datasets.path_helper import expand_path_to_data_dirs
from spine_segmentation.visualisation.blender.open_in_blender import open_in_blender


def main():
    from spine_segmentation.resources.other_paths import RAW_NAKO_DATASET_PATH

    inference = SegmentationInference()
    sample_iterator = SampleIterator(
        expand_path_to_data_dirs(RAW_NAKO_DATASET_PATH), add_adjacent_slices=True
    )

    for image, *_ in sample_iterator:
        print(image.shape)
        assert image.shape[1] == 3
        segmentation = inference.segment(image[:, 1, :, :], batch_size=1)
        open_in_blender({"instances": segmentation.instance_segmentation})
        break


if __name__ == "__main__":
    main()

from pathlib import Path

from spine import SegmentationInference
from spine.datasets.sample import SampleIterator
from spine.datasets.segmentation_dataset import expand_path_to_data_dirs
from spine.visualisation.blender.open_in_blender import open_in_blender


def main():
    from spine.resources.other_paths import RAW_NAKO_DATASET_PATH

    inference = SegmentationInference()
    sample_iterator = SampleIterator(
        expand_path_to_data_dirs(RAW_NAKO_DATASET_PATH), add_adjacent_slices=True, skip_first_percent=0.9505
    )

    for image, *_ in sample_iterator:
        print(image.shape)
        segmentation = inference.instance_segmentation_for_image(image)
        print(segmentation)
        open_in_blender({"instances": segmentation["instances_post_processed"]})
        break


if __name__ == "__main__":
    main()

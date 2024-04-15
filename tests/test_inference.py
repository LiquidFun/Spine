from pathlib import Path

import numpy as np

from spine_segmentation import SegmentationInference
from spine_segmentation.datasets.sample import SampleIterator
from spine_segmentation.datasets.path_helper import expand_path_to_data_dirs
from spine_segmentation.visualisation.blender.open_in_blender import open_in_blender


def test_inference_on_real_data():
    from spine_segmentation.resources.other_paths import RAW_NAKO_DATASET_PATH

    inference = SegmentationInference()
    sample_iterator = SampleIterator(
        expand_path_to_data_dirs(RAW_NAKO_DATASET_PATH), add_adjacent_slices=True
    )

    for image, *_ in sample_iterator:
        print(image.shape)
        assert image.shape[1] == 3
        import matplotlib.pyplot as plt
        img = image[:, 1, :, :400]
        # plt.imshow(img)
        # plt.hist(img)
        # plt.show()
        segmentation = inference.segment(img, batch_size=1)
        print(segmentation.instance_segmentation.shape)
        open_in_blender({"instances": segmentation.instance_segmentation})
        break



def test_on_random_too_large():
    image = np.random.rand(20, 320, 1324)
    inference = SegmentationInference(output_same_shape_as_input=True)

    segmentation = inference.segment(image, batch_size=1)


if __name__ == '__main__':
    test_inference_on_real_data()
    # test_on_random_too_large()
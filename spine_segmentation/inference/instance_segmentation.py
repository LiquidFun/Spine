from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union, Dict

from loguru import logger

from spine_segmentation.inference.onnx_model import ONNXInferenceModel
from spine_segmentation.instance_separation.instance_separation import (
    get_planes_from_rois,
    mask_rare_rois,
    separate_rois_with_labels,
    separate_segmented_rois_by_planes,
)
from spine_segmentation.spine_types.labels import get_labels_for_n_classes
from spine_segmentation.utils.log_dir import get_next_log_dir
from spine_segmentation.visualisation.blender.open_in_blender import open_in_blender

import numpy as np


def post_process_instances(vertebra_instances, segmentation, plot_dir=None):
    vertebra_instances = vertebra_instances.copy()
    vertebra = segmentation == 1

    vertebra_instances[~vertebra] = 0
    split_segmentation, _ = separate_rois_with_labels(segmentation)
    split_vertebra_segmentation = split_segmentation.copy()
    split_vertebra_segmentation[~vertebra] = 0
    split_vertebra_segmentation[vertebra] -= np.unique(split_vertebra_segmentation)[1] - 1
    add2 = vertebra * 2
    print(split_vertebra_segmentation.shape, np.unique(split_vertebra_segmentation))
    add2_only_s1 = (split_vertebra_segmentation == split_vertebra_segmentation.max()) * 2

    best_vertebra_instances = split_vertebra_segmentation
    best_mse = 1e9

    plots = [(vertebra_instances, "Inst", np.zeros_like(vertebra_instances))]

    for ith, add_for_s1 in enumerate([0, add2_only_s1, add2_only_s1 * 2]):
        for i in range(26):
            curr_vertebra_instances = split_vertebra_segmentation + add2 * i + add_for_s1
            if (ith != 0 and curr_vertebra_instances.max() != 49) or curr_vertebra_instances.max() >= 50:
                continue
            # error_map = np.clip((vertebra_instances - curr_vertebra_instances) ** 2, -9, 9)
            error_map = np.abs(vertebra_instances - curr_vertebra_instances) + (
                vertebra_instances != curr_vertebra_instances
            )
            # error_map = vertebra_instances - curr_vertebra_instances
            # error_map[~vertebra] = 0
            mse = np.mean(error_map)
            plots.append((curr_vertebra_instances, f"{mse:.4f}", error_map))
            # print(
            #     f"MSE: {mse:.3f} {best_mse:.3f}",
            #     np.unique(curr_vertebra_instances),
            #     list(zip(*np.unique(vertebra_instances, return_counts=True))),
            # )
            if mse < best_mse:
                best_mse = mse
                best_vertebra_instances = curr_vertebra_instances


    # vertebra_instances = split_vertebra_segmentation + add2 * best_i
    vertebra_instances = best_vertebra_instances

    planes = get_planes_from_rois(vertebra_instances)
    # vertebrae = separate_segmented_rois(arr3d == 1)
    disc_instances = separate_segmented_rois_by_planes((segmentation == 2).astype(int), planes, use_plane_index=True)
    disc_instances[vertebra_instances != 0] = 0

    instances = vertebra_instances + disc_instances
    return mask_rare_rois(instances, 50)


def print_stats_summary(stats):
    correct_sum, correct_no_edges_sum = 0, 0
    for stat in stats:
        if stat:
            correct_sum += stat["correct"]
            correct_no_edges_sum += stat["correct_no_edges"]
    print(f"{correct_sum=} / {len(stats)}, {correct_no_edges_sum=} / {len(stats)}")


@dataclass
class SegmentationResult:
    semantic_segmentation: np.ndarray
    instance_segmentation: np.ndarray
    id_to_label: Dict[int, str]


DeviceType = Union[Literal["CPU", "GPU"], int]


def _add_channel_to_3d_mri(mri3d: np.ndarray):
    """Given a shape (20, 320, 896), convert it to (20, 3, 320, 896), where in the RGB channel adjacent slices are encoded"""
    first = np.roll(mri3d, axis=0, shift=1)
    first[0, :, :] = first[1, :, :]

    last = np.roll(mri3d, axis=0, shift=-1)
    last[-1, :, :] = last[-2, :, :]

    result = np.stack([first, mri3d, last], axis=1)
    new_shape = (mri3d.shape[0], 3, mri3d.shape[1], mri3d.shape[2])
    assert new_shape == result.shape
    return result


class SegmentationInference:

    def __init__(self, segmentation_device: DeviceType = "CPU", instance_segmentation_device: DeviceType = "CPU", *, output_same_shape_as_input: bool = False):
        self._model_image_height = 896
        self._model_image_width = 320
        self._onnx_seg_inference = ONNXInferenceModel.get_best_segmentation_model(device=segmentation_device)
        self._onnx_inst_inference = ONNXInferenceModel.get_best_instance_segmentation_model(
            device=instance_segmentation_device
        )
        self._output_same_shape_as_input = output_same_shape_as_input

    def segment(
        self,
        mri_3d_numpy_image: np.ndarray,
        *,
        batch_size: int,
        cache_dir: Union[Literal["auto"], None, Path, str] = "auto",
    ):
        initial_shape = mri_3d_numpy_image.shape
        if len(mri_3d_numpy_image.shape) < 3:
            logger.error(
                f"mri_3d_numpy_image must be 3 dimensional. Got {mri_3d_numpy_image.shape}. If you "
                f"wish to segment only a 2D slice, consider repeating the slice three times in the "
                f"first dimension, resulting in this shape: {(3, *mri_3d_numpy_image.shape)}:"
                "\n\n\tnp.tile(input_image, (3, 1, 1))\n"
            )
            exit(1)
        if len(mri_3d_numpy_image.shape) > 3:
            logger.error(f"mri_3d_numpy_image must be 3 dimensional. Got {mri_3d_numpy_image.shape}. "
                         f"Consider dropping the batch-dimension and just using the 3D mri image as entire input.")

        mri_3d_numpy_image = self._crop_and_pad_to_shape(mri_3d_numpy_image, self._model_image_width, self._model_image_height)
        mri_4d_with_channels = _add_channel_to_3d_mri(mri_3d_numpy_image)

        seg_npz = self._onnx_seg_inference.inference(mri_4d_with_channels)
        inst_seg_input = seg_npz["segmentation"][:, None, :, :]
        cropped_inst_seg_input = inst_seg_input[:, :, :, :self._model_image_height]
        cropped_input_image_with_gt = np.concatenate(
            [
                mri_4d_with_channels[:, 1:2, :, :],
                (cropped_inst_seg_input == 1),
                (cropped_inst_seg_input == 2),
            ],
            axis=1,
        ).astype(np.float32)
        cropped_input_image_without_gt = mri_4d_with_channels[:, :, :, :].astype(np.float32)

        new_npz = self._onnx_inst_inference.inference(cropped_input_image_without_gt)
        new_npz["instances"] = new_npz["instances"] * 2 - (new_npz["instances"] != 0)
        # new_npz["id_to_label"] = gt_npz["id_to_label"]
        # new_npz["gt_id_to_label"] = gt_npz["id_to_label"]
        new_npz["instances_post_processed"] = post_process_instances(
            new_npz["instances"], cropped_inst_seg_input[:, 0, :, :], plot_dir=cache_dir
        )
        new_npz["cropped_segmentation"] = cropped_inst_seg_input[:, 0, :, :]
        id_to_labels = get_labels_for_n_classes(49)

        instances = new_npz["instances_post_processed"]
        segmentation = new_npz["segmentation"]
        if self._output_same_shape_as_input:
            segmentation = self._crop_and_pad_to_shape(segmentation, initial_shape[1], initial_shape[2])
            instances = self._crop_and_pad_to_shape(instances, initial_shape[1], initial_shape[2])
            assert segmentation.shape == initial_shape

        return SegmentationResult(segmentation, instances, id_to_labels)

    def _crop_and_pad_to_shape(self, mri_3d_with_channels: np.ndarray, target_width, target_height):
        _, width, height = mri_3d_with_channels.shape

        # Crop
        from_width = width // 2 - target_width // 2
        to_width = from_width + target_width
        mri_3d_with_channels = mri_3d_with_channels[:, from_width:to_width, 0:target_height]

        # Padding
        _, width, height = mri_3d_with_channels.shape
        # 2 different widths, because they could differ by 1 pixel
        half_width_padding = (target_width - width) // 2
        other_half_padding = target_width - width - half_width_padding
        height_padding = target_height - height
        padding_to_896 = ((0, 0), (half_width_padding, other_half_padding), (0, height_padding))
        mri_3d_with_channels = np.pad(mri_3d_with_channels, padding_to_896, mode="constant")
        assert mri_3d_with_channels.shape[1:] == (target_width, target_height), f"{mri_3d_with_channels.shape[1:]} != {(target_width, target_height)=}"
        return mri_3d_with_channels.astype(np.float32)

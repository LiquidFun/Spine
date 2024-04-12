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


def get_dataloader(
    train_or_val: Literal["train", "val"], path: Union[Path, str] = None, *, bs: int = 16, crop_height: int = None
):
    from spine_segmentation.datasets.segmentation_dataset import SegmentationDataModule

    args = [] if path is None else [path]
    data_module = SegmentationDataModule(
        *args,
        # add_bs_wk_as_channels=True,
        add_adjacent_slices=True,
        batch_size=bs,
        num_workers=4,
        crop_height_to_px=crop_height,
        target_shape=(None, 320, 896),
    )
    if train_or_val == "train":
        return data_module.train_dataloader()
    elif train_or_val == "val":
        return data_module.val_dataloader()


def plot_model(onnx_path):
    import netron

    netron.start(onnx_path)


def open_npz_in_blender(numpy_path):
    if isinstance(numpy_path, str) or isinstance(numpy_path, Path):
        spines = np.load(numpy_path)
    else:
        spines = numpy_path
    output = spines["instances_post_processed"]
    open_in_blender({"instances": output})


def open_npz_in_blender_separate_rois(npz, render_to_instead=None):
    # output, gt, instances = npz["segmentation"], npz["gt"], npz["instances"]
    # id_to_label = spines["id_to_label"].item()
    # discs = (instances % 2 == 0) * (instances != 0) * 2
    # separated_discs = separate_segmented_rois(discs) * 2
    # planes = get_planes_from_discs(separated_discs, id_to_label=id_to_label)
    # collection = BlenderCollection("planes", planes)

    open_in_blender({"instances": npz["instances"]}, render_to_instead=render_to_instead)
    # open_in_blender({"instances": instances, "gt": gt}, [collection], smooth_spine=False)


def open_npz_and_plot_rois(npz, plot_path):
    # spines = np.load(numpy_path)
    # output, gt, instances = spines["output"], spines["gt"], spines["instances"]
    # vertebrae = separate_segmented_rois(output == 1)
    # discs = separate_segmented_rois(output == 2)
    # instances = vertebrae * 2 + (discs * 2 + (discs != 0))
    plot_npz(npz, plot_path, slices=range(0, 16))


# def find_best_matching(vertebra_instances, vertebra_segmentation):


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

    if plot_dir is not None:
        import matplotlib.pyplot as plt

        # fig, axes = plt.subplots(4, (len(plots) + 3) // 4)
        fig, axes = plt.subplots(2, len(plots))
        fig.tight_layout(pad=0)

        from spine_segmentation.visualisation.color_palettes import get_alternating_palette

        id_to_labels = dict(zip(range(1, 50), get_labels_for_n_classes(49)))
        directory = get_next_log_dir()
        for i, (instances, title, error_map) in enumerate(plots):
            alternating = get_alternating_palette()
            ax = axes[0, i]
            single_plot(ax, alternating, None, instances[8, :, ::-1].T, id_to_labels, title, font_scale=0.4)
            ax = axes[1, i]
            single_plot(ax, alternating, error_map[8, :, ::-1].T != 0, None, id_to_labels, None, font_scale=0.4)

        plt.savefig(plot_dir / f"mse_comparison.png", dpi=600)
        plt.close()
    # fig.close()

    # np.unique(split_vertebra_segmentation + add2 * 100)
    # plt.imshow((vertebra_instances - (split_vertebra_segmentation + add2 * i))[10])
    # plt.imshow((split_vertebra_segmentation)[10])
    # plt.show()

    # vertebra_instances = split_vertebra_segmentation + add2 * best_i
    vertebra_instances = best_vertebra_instances

    # print(f"{np.unique(split_segmentation)=}")
    # for i in np.unique(split_segmentation)[1:]:
    #     instance_mask = split_segmentation == i
    #     instance = vertebra_instances[instance_mask]
    #     most_common = np.bincount(instance).argmax()
    #     vertebra_instances[instance_mask] = most_common

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


MODEL_HEIGHT = 896

DeviceType = Union[Literal["CPU", "GPU"], int]


class SegmentationInference:

    def __init__(self, segmentation_device: DeviceType = "CPU", instance_segmentation_device: DeviceType = "CPU"):
        self._model_height = MODEL_HEIGHT
        self._onnx_seg_inference = ONNXInferenceModel.get_best_segmentation_model(device=segmentation_device)
        self._onnx_inst_inference = ONNXInferenceModel.get_best_instance_segmentation_model(
            device=instance_segmentation_device
        )

    def segment(
        self,
        mri_3d_numpy_image: np.ndarray,
        *,
        batch_size: int,
        cache_dir: Union[Literal["auto"], None, Path, str] = "auto",
    ):
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

        padding_to_896 = ((0, 0), (0, 0), (0, 0), (0, mri_3d_numpy_image.shape[3] - MODEL_HEIGHT))
        cropped_padded_seg_input = np.pad(mri_3d_numpy_image, padding_to_896, mode="constant")
        seg_npz = self._onnx_seg_inference.inference(cropped_padded_seg_input)
        inst_seg_input = seg_npz["segmentation"][:, None, :, :]
        cropped_inst_seg_input = inst_seg_input[:, :, :, :MODEL_HEIGHT]
        padding_to_416 = ((0, 0), (0, 0), (0, 0), (0, MODEL_HEIGHT - MODEL_HEIGHT))
        cropped_input_image_with_gt = np.concatenate(
            [
                np.pad(mri_3d_numpy_image[:, 1:2, :, :], padding_to_416, mode="constant"),
                (cropped_inst_seg_input == 1),
                (cropped_inst_seg_input == 2),
            ],
            axis=1,
        )
        cropped_input_image_with_gt = cropped_input_image_with_gt.astype(np.float32)
        cropped_input_image_without_gt = np.pad(mri_3d_numpy_image[:, :, :, :], padding_to_416, mode="constant").astype(
            np.float32
        )

        new_npz = self._onnx_inst_inference.inference(cropped_input_image_without_gt)
        new_npz["instances"] = new_npz["instances"] * 2 - (new_npz["instances"] != 0)
        # new_npz["id_to_label"] = gt_npz["id_to_label"]
        # new_npz["gt_id_to_label"] = gt_npz["id_to_label"]
        new_npz["instances_post_processed"] = post_process_instances(
            new_npz["instances"], cropped_inst_seg_input[:, 0, :, :], plot_dir=cache_dir
        )
        new_npz["cropped_segmentation"] = cropped_inst_seg_input[:, 0, :, :]
        id_to_labels = get_labels_for_n_classes(49)
        return SegmentationResult(inst_seg_input, new_npz["instances_post_processed"], id_to_labels)

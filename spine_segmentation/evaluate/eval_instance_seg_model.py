import random
from pathlib import Path
from typing import Literal, Union

import torch

from spine_segmentation.cli.cli import colored_tracebacks
from spine_segmentation.datasets.sample import SampleIterator, get_random_crop_slice
from spine_segmentation.inference.onnx_model import ONNXInferenceModel
from spine_segmentation.instance_separation.instance_separation import (
    get_planes_from_rois,
    mask_rare_rois,
    separate_rois_with_labels,
    separate_segmented_rois_by_planes,
)
from spine_segmentation.plotting.plot_slice import get_stats, plot_npz, single_plot
from spine_segmentation.resources.paths import TRAIN_SPLIT_CSV_PATH, VAL_SPLIT_CSV_PATH
from spine_segmentation.resources.other_paths import RAW_NAKO_DATASET_PATH
from spine_segmentation.spine_types.labels import get_labels_for_n_classes
from spine_segmentation.utils.log_dir import get_next_log_dir
from spine_segmentation.visualisation.blender.open_in_blender import open_in_blender

colored_tracebacks()

import numpy as np

from spine_segmentation.datasets.segmentation_dataset import SegmentationDataModule
from spine_segmentation.datasets.path_helper import expand_path_to_data_dirs


def get_dataloader(
    train_or_val: Literal["train", "val"], path: Union[Path, str] = None, *, bs: int = 16, crop_height: int = None
):
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
    # from matplotlib import pyplot as plt

    # plt.imshow(vertebra_instances[8])
    # plt.imshow(instance_mask[8])
    # plt.imshow(vertebra[8])
    # plt.imshow(split_segmentation[8])
    # plt.show()
    vertebra_instances[~vertebra] = 0
    split_segmentation, _ = separate_rois_with_labels(segmentation)
    split_vertebra_segmentation = split_segmentation.copy()
    split_vertebra_segmentation[~vertebra] = 0
    split_vertebra_segmentation[vertebra] -= np.unique(split_vertebra_segmentation)[1] - 1
    add2 = vertebra * 2
    print(split_vertebra_segmentation.shape, np.unique(split_vertebra_segmentation))
    add2_only_s1 = (split_vertebra_segmentation == split_vertebra_segmentation.max()) * 2

    # best_i = 0
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


def plot_every_val_dataset(directory=None, device=3, model_height=896, cropped_height=None, cropped_V=None):
    if directory is None:
        directory = get_next_log_dir()
    directory = Path(directory)

    if cropped_V is not None:
        assert 1 <= cropped_V <= 49

    if cropped_height is not None:
        assert 50 <= cropped_V

    onnx_seg_inference = ONNXInferenceModel.get_best_segmentation_model(device=device)
    # onnx_seg_inference.load_index_list()

    onnx_inst_inference = ONNXInferenceModel.get_best_instance_segmentation_model(device=device)

    sample_iterator = SampleIterator(
        expand_path_to_data_dirs(RAW_NAKO_DATASET_PATH), add_adjacent_slices=True, skip_first_percent=0.9505
    )

    # index_list_path = (
    #     Path.home() / "devel/src/git/spine_annotator/cache/onnx_inference/2023-09-12_Seg_Unet_mitb5/index_list.npz"
    # )
    # val_dataloader = get_dataloader("val", index_list_path, bs=1)

    val_dataloader = SegmentationDataModule(
        [TRAIN_SPLIT_CSV_PATH, VAL_SPLIT_CSV_PATH],
        add_adjacent_slices=True,
        batch_size=16,
        # crop_height_to_px=416,
        target_shape=(18, 320, 896),
    ).val_dataloader()

    stats = []

    gt = None
    # cropped_height = 50
    random.seed(0)
    # for i, (image, _, sample, path) in enumerate(sample_iterator):
    # for i, (image, *_) in enumerate(sample_iterator):
    for i, (image, gt) in enumerate(val_dataloader):
        # if i >= 100:
        # break

        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()

        if gt is None:
            gt_npz = onnx_seg_inference.inference(image)
        else:
            gt_instances, id_to_label = separate_rois_with_labels(gt.numpy()[:, 0, :, :])
            gt_npz = {"instances": gt_instances, "id_to_label": id_to_label}

        gt_npz["instances"][gt_npz["instances"] == gt_npz["instances"].max()] = 49

        print(gt_npz["instances"].shape)

        orig_gt_nonzero = np.nonzero(np.any(gt_npz["instances"], axis=(0, 1)))[0]
        gt_start_at = orig_gt_nonzero[0]
        gt_end_at = orig_gt_nonzero[-1]
        print(f"{gt_start_at=} {gt_end_at=}")
        how_many_crops = 10
        if cropped_height == model_height == 896:
            how_many_crops = 1
            gt_end_at = gt_start_at = 0

        # for start_at in np.linspace(gt_start_at, gt_end_at - cropped_height, how_many_crops):
        for start_at_class in range(1, 49 - cropped_V + 2, 2):  # +2 so that 49 is inclusive
            end_at_class = start_at_class + cropped_V - 1
            try:
                print(f"\n\n============ Processing sample {i}: {start_at_class}-{end_at_class} =============\n")
                print(gt_npz["instances"].shape)
                relevant_classes = np.any(
                    (gt_npz["instances"] >= start_at_class) & (gt_npz["instances"] <= end_at_class), axis=(0, 1)
                )
                print(f"{relevant_classes.shape=}")
                nonzero_relevant_classes = np.nonzero(relevant_classes)[0]
                start_at = nonzero_relevant_classes[0]
                end_at = nonzero_relevant_classes[-1]
                cropped_height = end_at - start_at
                print(f"{start_at=} {end_at=} {cropped_height=}")

                image = np.array(image)
                print(image.shape)

                crop_slice = get_random_crop_slice(cropped_height, image.shape[3], start_at=round(start_at))

                curr_dir = directory / f"sample_{i:03}_{crop_slice.start:03}-{crop_slice.stop:03}"
                curr_dir.mkdir(parents=True, exist_ok=True)

                # ==================
                # Create ground truth segmentation
                # ==================

                cropped_instance_gt = gt_npz["instances"][:, None, :, crop_slice].copy(order="C")

                # ==================
                # Create cropped input image for segmentation model (which will be used as input for instance segmentation)
                # ==================

                cropped_seg_input = image[:, :, :, crop_slice].copy(order="C")
                padding_to_896 = ((0, 0), (0, 0), (0, 0), (0, image.shape[3] - cropped_height))
                cropped_padded_seg_input = np.pad(cropped_seg_input, padding_to_896, mode="constant")
                seg_npz = onnx_seg_inference.inference(cropped_padded_seg_input)

                inst_seg_input = seg_npz["segmentation"][:, None, :, :]
                cropped_inst_seg_input = inst_seg_input[:, :, :, :model_height]
                padding_to_416 = ((0, 0), (0, 0), (0, 0), (0, model_height - cropped_height))
                cropped_input_image_with_gt = np.concatenate(
                    [
                        np.pad(image[:, 1:2, :, crop_slice], padding_to_416, mode="constant"),
                        (cropped_inst_seg_input == 1),
                        (cropped_inst_seg_input == 2),
                    ],
                    axis=1,
                )
                cropped_input_image_with_gt = cropped_input_image_with_gt.astype(np.float32)

                cropped_input_image_without_gt = np.pad(
                    image[:, :, :, crop_slice], padding_to_416, mode="constant"
                ).astype(np.float32)

                # ==================
                # Create cropped input image for instance segmentation model
                # ==================

                # cropped_instance_seg_input = full_input_image_with_gt[:, :, :, crop_slice].copy(order="C")

                # if True:
                #     padding = ((0, 0), (0, 0), (0, 0), (0, image.shape[3] - height))
                #     cropped_image = np.pad(cropped_image, padding, mode="constant")
                #     cropped_gt = np.pad(cropped_gt, padding, mode="constant")

                # npz["instances"] = onnx_inst_inference.inference(cropped_image, gt=cropped_gt)["instances"]
                # npz["gt_instances"] = npz["instances"]
                new_npz = onnx_inst_inference.inference(
                    cropped_input_image_without_gt, gt=np.pad(cropped_instance_gt, padding_to_416, mode="constant")
                )
                new_npz["instances"] = new_npz["instances"] * 2 - (new_npz["instances"] != 0)
                new_npz["id_to_label"] = gt_npz["id_to_label"]
                new_npz["gt_id_to_label"] = gt_npz["id_to_label"]

                new_npz["instances_post_processed"] = post_process_instances(
                    new_npz["instances"], cropped_inst_seg_input[:, 0, :, :], plot_dir=curr_dir
                )

                new_npz["cropped_segmentation"] = cropped_inst_seg_input[:, 0, :, :]

                stats.append(get_stats(new_npz["instances_post_processed"], new_npz["gt_instances"]))
                np.savez_compressed(directory / "stats.npz", stats=stats)

                open_npz_in_blender(new_npz)
                plot_npz(new_npz, curr_dir / "plot.png", slices=range(7, 10), stats=stats[-1])
                print_stats_summary(stats)
                # if i > 50:
                #    break
                # exit(0)
            except KeyboardInterrupt:
                raise
            except:
                print(f"Error processing sample {i}")
                stats.append({})
                import traceback

                traceback.print_exc()
                continue

    # open_npz_and_plot_rois(new_npz, curr_dir / "plot.png")
    # open_npz_in_blender_separate_rois(npz, curr_dir / "render.png")


def main():
    # Get first argument as int
    import sys

    slice_height = int(sys.argv[1]) if len(sys.argv) > 1 else 49
    gpu = int(sys.argv[2]) if len(sys.argv) > 2 else "CPU"
    # plot_every_val_dataset(cropped_height=slice_height, device=gpu)
    plot_every_val_dataset(cropped_V=slice_height, device=gpu)


if __name__ == "__main__":
    main()

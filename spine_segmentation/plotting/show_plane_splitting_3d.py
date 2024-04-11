import random

from spine_segmentation.cli.cli import colored_tracebacks
from spine_segmentation.datasets.sample import get_random_crop_slice
from spine_segmentation.evaluate.eval_instance_seg_model import post_process_instances
from spine_segmentation.inference.onnx_model import ONNXInferenceModel
from spine_segmentation.plotting.plot_slice import get_stats, plot_npz
from spine_segmentation.resources.paths import TRAIN_SPLIT_CSV_PATH, VAL_SPLIT_CSV_PATH
from spine_segmentation.utils.log_dir import get_next_log_dir

colored_tracebacks()

import numpy as np

from spine_segmentation.datasets.segmentation_dataset import SegmentationDataModule


def eval_every_seg_sample_by_splitting(directory=None, device=3, height=416):
    if directory is None:
        directory = get_next_log_dir()

    onnx_seg_inference = ONNXInferenceModel.get_best_segmentation_model(device=device)
    # onnx_seg_inference.load_index_list()

    # onnx_inst_inference = ONNXInferenceModel.get_best_instance_segmentation_model(px=height, device=device)

    # sample_iterator = SampleIterator(
    #    expand_path_to_data_dirs(RAW_NAKO_DATASET_PATH), add_adjacent_slices=True, skip_first_percent=0.9505
    # )

    # index_list_path = (
    #    Path.home() / "devel/src/git/spine_annotator/cache/onnx_inference/2023-09-12_Seg_Unet_mitb5/index_list.npz"
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

    random.seed(0)
    # for i, (image, _) in enumerate(val_dataloader):
    for i, (image, _) in enumerate(val_dataloader):
        try:
            print(f"\n\n==================== Processing sample {i} =====================\n")
            image = np.array(image)
            print(image.shape)

            curr_dir = directory / f"sample_{i}"
            curr_dir.mkdir(parents=True, exist_ok=True)

            cropped_height = 320
            crop_slice = get_random_crop_slice(cropped_height, image.shape[3])

            # ==================
            # Create ground truth segmentation
            # ==================

            gt_npz = onnx_seg_inference.inference(image)
            # gt = gt_npz["segmentation"][:, None, :, :]
            cropped_instance_gt = gt_npz["instances"][:, None, :, crop_slice].copy(order="C")

            # ==================
            # Create cropped input image for segmentation model (which will be used as input for instance segmentation)
            # ==================

            cropped_seg_input = image[:, :, :, crop_slice].copy(order="C")
            padding_to_896 = ((0, 0), (0, 0), (0, 0), (0, image.shape[3] - cropped_height))
            cropped_padded_seg_input = np.pad(cropped_seg_input, padding_to_896, mode="constant")
            seg_npz = onnx_seg_inference.inference(cropped_padded_seg_input)

            inst_seg_input = seg_npz["segmentation"][:, None, :, :]
            cropped_inst_seg_input = inst_seg_input[:, :, :, :height]
            padding_to_416 = ((0, 0), (0, 0), (0, 0), (0, height - cropped_height))
            cropped_input_image_with_gt = np.concatenate(
                [
                    np.pad(image[:, 1:2, :, crop_slice], padding_to_416, mode="constant"),
                    (cropped_inst_seg_input == 1),
                    (cropped_inst_seg_input == 2),
                ],
                axis=1,
            )
            cropped_input_image_with_gt = cropped_input_image_with_gt.astype(np.float32)

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
                cropped_input_image_with_gt, gt=np.pad(cropped_instance_gt, padding_to_416, mode="constant")
            )
            new_npz["instances"] = new_npz["instances"] * 2 - (new_npz["instances"] != 0)
            new_npz["id_to_label"] = gt_npz["id_to_label"]
            new_npz["gt_id_to_label"] = gt_npz["id_to_label"]

            new_npz["instances_post_processed"] = post_process_instances(
                new_npz["instances"], cropped_inst_seg_input[:, 0, :, :]
            )

            new_npz["cropped_segmentation"] = cropped_inst_seg_input[:, 0, :, :]

            stats.append(get_stats(new_npz["instances_post_processed"], new_npz["gt_instances"]))
            np.savez_compressed(directory / "stats.npz", stats=stats)

            plot_npz(new_npz, curr_dir / "plot.png", slices=range(0, 16))
            # if i > 50:
            #    break
            # exit(0)
        except KeyboardInterrupt:
            raise
        except:
            print(f"Error processing sample {i}")
            stats.append("Error")
            import traceback

            traceback.print_exc()
            continue

    # open_npz_and_plot_rois(new_npz, curr_dir / "plot.png")
    # open_npz_in_blender_separate_rois(npz, curr_dir / "render.png")


def main():
    eval_every_seg_sample_by_splitting()


if __name__ == "__main__":
    main()

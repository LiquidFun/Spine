import concurrent.futures
import random
from concurrent.futures import ProcessPoolExecutor

from spine_segmentation.cli.cli import colored_tracebacks
from spine_segmentation.datasets.sample import get_random_crop_slice
from spine_segmentation.inference.onnx_model import ONNXInferenceModel
from spine_segmentation.instance_separation.instance_separation import separate_rois_with_labels
from spine_segmentation.plotting.plot_slice import get_stats, plot_npz
from spine_segmentation.utils.log_dir import get_next_log_dir

colored_tracebacks()

import numpy as np


def is_invalid(inst: np.ndarray) -> bool:
    # print(inst)
    uniq = np.unique(inst)[1:]
    if min(uniq) != 1:
        print("min", min(uniq))
        return True
    if max(uniq) not in [45, 47, 49]:
        print("max", max(uniq))
        return True
    if len(uniq) not in [45, 47, 49]:
        print("len", len(uniq))
        return True
    odd = np.sum((uniq % 2) == 1)
    even = np.sum((uniq % 2) == 0)
    if odd not in [23, 24, 25]:
        print("odd", odd)
        return True
    if even not in [22, 23, 24]:
        print("even", even)
        return True
    return False


def process_npz(i, index, npz_path, directory):
    # if i % 100 == 0:
    #    print(i)
    npz = dict(np.load(npz_path, allow_pickle=True))
    # print(dict(npz).keys())
    connected_components_inst = separate_rois_with_labels(npz["segmentation"], split_along_discs=False)[0]
    split_by_discs_inst = npz["instances"]
    cc_invalid = is_invalid(connected_components_inst)
    sd_invalid = is_invalid(split_by_discs_inst)
    if cc_invalid or sd_invalid:
        print(f"\tInvalid sample {i=} cc:{cc_invalid} sd:{sd_invalid}")
        npz["original"] = connected_components_inst[:, None, :, :]
        npz["gt_instances"] = connected_components_inst
        plot_npz(npz, directory / f"sample_{i}_{cc_invalid}_{sd_invalid}.png")
    return cc_invalid, sd_invalid, index, i


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

    with ProcessPoolExecutor(max_workers=120) as executor:
        futures = []
        for i, (index, npz_path) in enumerate(onnx_seg_inference.index_to_npz_path.items()):
            # process_npz(i, index, npz_path, directory)
            futures.append(executor.submit(process_npz, i, index, npz_path, directory))

        cc_invalids = 0
        sd_invalids = 0
        for i_as_completed, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                res = future.result()
                cc_invalids += res[0]
                sd_invalids += res[1]
                # index = res[2]
                # i_orig = res[3]
                print(f"{i_as_completed=} {cc_invalids=} {sd_invalids=}")
                # npz_path = onnx_seg_inference.index_to_npz_path[index]
            except Exception as exc:
                print(f"generated an exception: {exc}")

        # for future in concurrent.futures.as_completed(future_to_npz):
        #     index = future_to_npz[future]
        #     try:
        #         if future.result():
        #             break
        #     except Exception as exc:
        #         print(f'{index} generated an exception: {exc}')

    # Make this parallel
    # for index, npz_path in onnx_seg_inference.index_to_npz_path.items():
    #     print(index, npz_path)
    #     if index > 1000:
    #         break
    return
    stats = []

    random.seed(0)
    # for i, (image, _) in enumerate(val_dataloader):
    for i, (image, _, sample, path) in enumerate(sample_iterator):
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

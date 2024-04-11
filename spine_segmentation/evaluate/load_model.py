from pathlib import Path
from typing import Literal, Union

from spine_segmentation.cli.cli import colored_tracebacks
from spine_segmentation.datasets.sample import SampleIterator
from spine_segmentation.inference.onnx_model import ONNXInferenceModel
from spine_segmentation.plotting.plot_slice import plot_npz
from spine_segmentation.resources.paths import NAKO_DATASET_PATH
from spine_segmentation.utils.log_dir import get_next_log_dir
from spine_segmentation.visualisation.blender.open_in_blender import open_in_blender

colored_tracebacks()

import numpy as np
import onnx
from loguru import logger

from spine_segmentation.datasets.segmentation_dataset import SegmentationDataModule
from spine_segmentation.datasets.path_helper import expand_path_to_data_dirs

# onnx_model_path = "models/segmentation/2023-08-31_Seg_Unet_mit.onnx"
onnx_model_path = "models/segmentation/2023-09-12_Seg_Unet_mitb5.onnx"


def get_dataloader(train_or_val: Literal["train", "val"], path: Union[Path, str] = None, *, bs: int = 16):
    args = [] if path is None else [path]
    data_module = SegmentationDataModule(*args, augment=False, add_adjacent_slices=True, batch_size=bs, num_workers=4)
    if train_or_val == "train":
        return data_module.train_dataloader()
    elif train_or_val == "val":
        return data_module.val_dataloader()


def get_indices_from_train_dataloader(indices):
    train_data_loader = get_dataloader("train")
    samples = [sample for i, sample in enumerate(train_data_loader) if i in indices]
    return samples


def get_samples_with_l4():
    return get_indices_from_train_dataloader({13})


def get_samples_with_l6():
    return get_indices_from_train_dataloader({9, 10})


def get_sample(bs=16):
    data = get_dataloader("val", bs=bs)
    # Shape: (N, C, H, W): (16, 3, 320, 896), (16, 1, 320, 896)
    sample = next(iter(data))
    return sample


def resave_with_batch_size(onnx_path, save_to, new_batch_size=1):
    # Load the ONNX model
    model = onnx.load(onnx_path)

    # Get the name of the first input of the model
    # input_name = model.graph.input[0].name

    # Change the model's first input to the desired batch size
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = new_batch_size
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = new_batch_size

    # Save the modified model
    onnx.save(model, save_to)


def plot_model(onnx_path):
    import netron

    netron.start(onnx_path)


def open_npz_in_blender(numpy_path):
    spines = np.load(numpy_path)
    output, gt = spines["output"], spines["gt"]
    open_in_blender({"output": output, "gt": gt})


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


def render_all_samples(directory=None):
    if directory is None:
        directory = get_next_log_dir()
    directory = Path(directory)

    onnx_inference = ONNXInferenceModel.get_best_segmentation_model()
    # onnx_inference.load_index_list()
    # already_processed = set(onnx_inference.index_to_zip_path.values())
    already_processed = set()

    # for i, sample in enumerate(get_dataloader("train", RAW_NAKO_DATASET_PATH)):
    sample_iterator = SampleIterator(
        expand_path_to_data_dirs(NAKO_DATASET_PATH), add_adjacent_slices=True, already_processed=already_processed
    )

    try:
        # for i, (image, gt, sample, path) in enumerate(sample_iterator):
        for i, sample in enumerate(get_dataloader("train", NAKO_DATASET_PATH)):
            if i > 20:
                break
            # try:
            print(f"\n\n==================== Processing sample {i} {onnx_inference.index=} =====================\n")

            curr_dir = directory / f"sample_{i}"
            curr_dir.mkdir(parents=True, exist_ok=True)

            # sample_thing, image, gt = sample[0], sample[1]
            image, gt = sample[0].cpu().detach().numpy(), sample[1].cpu().detach().numpy()
            print(image.shape, gt.shape)

            # if sample.dicom is not None:
            # onnx_inference.index_to_series_id[onnx_inference.index] = sample.dicom.SeriesInstanceUID
            # onnx_inference.index_to_patient_id[onnx_inference.index] = sample.dicom.PatientID
            # onnx_inference.index_to_zip_path[onnx_inference.index] = sample.path

            npz = onnx_inference.inference(image, gt=gt)

            # if i % 100 == 0:
            #     onnx_inference.save_index_list()
            # except:
            #     print(f"Error processing sample {i}")
            #     continue
            npz["segmentation"] = npz["gt"].astype(int)
            # npz["instances"] = npz["gt_instances"]
            open_npz_and_plot_rois(npz, curr_dir / "plot.png")
            open_npz_in_blender_separate_rois(npz, curr_dir / "render.png")
            # open_npz_in_blender_separate_rois(npz)
    except KeyboardInterrupt:
        # print("Saving index list...")
        pass
    except:
        print("Error")
        import traceback

        traceback.print_exc()
    #    onnx_inference.save_index_list()


@logger.catch
def main():
    reuse_path = "logs/2023-09-12_154326"
    render_all_samples()


if __name__ == "__main__":
    main()

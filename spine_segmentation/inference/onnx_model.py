from functools import lru_cache
from pathlib import Path
from typing import Dict, Literal, Union

import numpy as np
import onnxruntime
import xxhash

from spine_segmentation.instance_separation.instance_separation import separate_rois_with_labels
from spine_segmentation.model_downloader.model_downloader import download_file_with_progress
from spine_segmentation.resources.paths import CACHE_PATH, MODELS_PATH
from spine_segmentation.utils.profiling import profile


class ONNXInferenceModel:
    def __init__(
        self,
        onnx_model_path: Union[Path, str],
        is_segmentation_model: bool,
        device: Union[Literal["CPU"], int] = "CPU",
        height: int = 896,
    ):
        self.onnx_model_path = Path(onnx_model_path)
        if not self.onnx_model_path.exists():
            raise FileNotFoundError(f"Model {self.onnx_model_path.absolute()} does not exist!")
        self.device = device
        self.cache_path = CACHE_PATH / "onnx_inference" / self.onnx_model_path.stem
        self.index_list = None
        if (self.cache_path / "index_list.npz").exists():
            self.load_index_list()

        self.height = height

        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.is_segmentation_model = is_segmentation_model
        self.index = 0

        self.index_to_npz_path: Dict[int, str] = {}
        self.index_to_slice_index: Dict[int, int] = {}
        self.index_to_series_id = {}
        self.index_to_patient_id = {}
        self.index_to_zip_path = {}

    @staticmethod
    def get_best_segmentation_model(*args, **kwargs):
        model_url = "https://github.com/LiquidFun/Spine/releases/download/onnx_models_1.0.0/2023-11-06_Seg2_Unet_resnet152_896px.onnx"
        onnx_model_path = download_file_with_progress(model_url)
        return ONNXInferenceModel(onnx_model_path, *args, **kwargs, is_segmentation_model=True)

    @staticmethod
    def get_best_instance_segmentation_model(*args, **kwargs):
        model_url = "https://github.com/LiquidFun/Spine/releases/download/onnx_models_1.0.0/2023-11-09_Seg25_Unet_resnet152_vert-only_896px_no-input-seg.onnx"
        onnx_model_path = download_file_with_progress(model_url)
        return ONNXInferenceModel(onnx_model_path, *args, **kwargs, is_segmentation_model=False)

    def save_index_list(self):
        path = self.cache_path / "index_list.npz"
        np.savez_compressed(
            path,
            index_to_npz_path=self.index_to_npz_path,
            index_to_slice_index=self.index_to_slice_index,
            index_to_series_id=self.index_to_series_id,
            index_to_patient_id=self.index_to_patient_id,
            index_to_zip_path=self.index_to_zip_path,
        )

    def load_index_list(self):
        path = self.cache_path / "index_list.npz"
        index_list = np.load(path, allow_pickle=True)
        self.index = max(index_list["index_to_slice_index"].item()) + 1
        self.index_to_npz_path = index_list["index_to_npz_path"].item()
        self.index_to_slice_index = index_list["index_to_slice_index"].item()
        self.index_to_series_id = index_list["index_to_series_id"].item()
        self.index_to_patient_id = index_list["index_to_patient_id"].item()
        self.index_to_zip_path = index_list["index_to_zip_path"].item()

    @property
    @lru_cache(maxsize=1)
    def _onnx_session(self):
        device_type = "CPU" if self.device == "CPU" else "CUDA"
        # options = onnxruntime.SessionOptions()
        session_kwargs = {}
        if device_type == "CUDA":
            assert isinstance(self.device, int)
            session_kwargs["provider_options"] = [{"device_id": self.device}]
        return onnxruntime.InferenceSession(
            self.onnx_model_path, providers=[f"{device_type}ExecutionProvider"], **session_kwargs
        )

    def _hash_array(self, array: np.ndarray) -> str:
        # Make array c contiguous
        # array = np.ascontiguousarray(array)
        return xxhash.xxh128(array).hexdigest()

    def _get_cached_npz_path(self, array: np.ndarray) -> Union[None, Path]:
        hashed = self._hash_array(array)
        path = self.cache_path / "output" / f"{hashed}.npz"
        print(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @profile
    def inference(self, image: np.ndarray, *, gt: np.ndarray = None, cache: bool = True):
        """
        :param image: shape (batch_size, channels, height, width)
        :param gt: ground truth, shape (batch_size, height, width) (OPTIONAL: only used to save in cache)
        :param cache: whether to retrieve from cache if already exists
        """
        if gt is not None:
            assert gt.shape[1] == 1
            gt = gt[:, 0, :, :]
            assert len(gt.shape) == 3, f"{gt.shape=}"
            assert gt.shape == image[:, 0, :, :].shape
        assert len(image.shape) == 4, f"{image.shape=}"

        # if image.shape[3] != self.height:
        #     padding = ((0, 0), (0, 0), (0, 0), (0, self.height - image.shape[2]))
        #     image = np.pad(image, padding, mode="constant")

        cached_path = self._get_cached_npz_path(image)

        if not cached_path.exists() or not cache:
            # assert False
            input_name = self._onnx_session.get_inputs()[0].name
            onnx_inputs = {input_name: image}
            onnx_outputs = self._onnx_session.run(None, onnx_inputs)
            onnx_outputs = np.array(onnx_outputs)

            segmentation = np.argmax(onnx_outputs[0], axis=1)

            if self.is_segmentation_model:
                instances, id_to_label = separate_rois_with_labels(segmentation)
                save_arrays = dict(
                    # original=image,
                    segmentation=segmentation,
                    instances=instances,
                    id_to_label=id_to_label,
                )

                assert (
                    image[:, 0, :, :].shape == segmentation.shape == instances.shape
                ), f"{segmentation.shape=} != {instances.shape=} != {image[:, 0, :, :].shape=}"

            else:
                save_arrays = dict(instances=segmentation)

            np.savez_compressed(cached_path, **save_arrays)

        npz = np.load(cached_path, allow_pickle=True)

        self.index_to_npz_path[self.index] = cached_path
        for i in range(len(npz["instances"])):
            self.index_to_slice_index[self.index] = i
            self.index += 1

        print(f"{len(self.index_to_npz_path)=} {len(self.index_to_slice_index)=}")

        npz_dict = dict(npz)
        npz_dict["original"] = image
        if gt is not None:
            key = "gt" if self.is_segmentation_model else "gt_instances"
            npz_dict[key] = gt
        return npz_dict

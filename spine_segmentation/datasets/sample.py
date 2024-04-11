import random
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Set, Tuple, Union

import nibabel as nib
import numpy as np
import pydicom

from spine_segmentation.instance_separation.instance_separation import separate_rois_with_labels
from spine_segmentation.resources.preloaded import get_measure_statistics

MetadataType = Literal["age", "sex", "weight", "size"]
MetadataTypes = Union[MetadataType, Iterable[MetadataType]]


@dataclass
class GTFormat:
    """The format of the ground truth.

    include: Which ground truth to include in the segmentation (wirbelkörper, bandscheiben).
    order: How to order the classes (any, y-sort), with y-sort being where the classes are ordered by
           their y-coordinate from up to down.
    separation: How to separate the classes (binary, semantic, instance), with instance each
                instance of a class is a separate class.
    type: The type of the ground truth (segmentation, classification, regression).

    Note that any other type than "segmentation" implies instance separation as well as y-sort
    """

    include: Literal["wk", "bs", "bs+wk"] = "bs+wk"
    order: Literal["any", "y-sort"] = "any"
    separation: Literal["binary", "semantic", "instance"] = "semantic"
    type: Literal["segmentation", "classification", "single-classification", "regression"] = "segmentation"
    s1_class: Optional[int] = None


def _normalize_image(img, target_shape, is_gt=False, return_nifti=False, return_full=False, crop_height_to_slice=None):
    if target_shape[0] is None:
        target_shape = (img.shape[0], *target_shape[1:])

    # Pad until target shape
    padding = np.subtract(target_shape, img.shape)
    padding[padding < 0] = 0
    img = np.pad(img, list(zip([0, 0, 0], padding)), mode="constant")
    assert img.shape >= target_shape, f"{img.shape=} {target_shape=}"

    # Reduce until target shape
    cx, cy, _ = [size // 2 for size in img.shape]
    sx, sy, sz = [s // 2 for s in target_shape]
    oddx, oddy, oddz = [s % 2 for s in target_shape]
    img = img[cx - sx : cx + sx + oddx, cy - sy : cy + sy + oddy, : sz * 2 + oddz]
    assert img.shape == target_shape, f"{img.shape=} {target_shape=}"

    # Normalize size
    if is_gt:
        img = (img != 0).astype(int)
        assert max(np.unique(img)) == img.max(), f"{np.unique(img)=} {img.max()=}"
    else:
        img = img.astype(np.float32)
        img -= img.min()
        img /= img.max()
        assert img.min() == 0 and img.max() == 1, f"{img.min()=} {img.max()=}"

    # print(img.shape, crop_height_to_slice)
    full_img = img
    if crop_height_to_slice is not None:
        img = img[:, :, crop_height_to_slice]
    # print(img.shape)

    if return_full:
        return img, full_img

    return img


def _load_and_normalize(path, *args, return_nifti=False, **kwargs):
    nifti = nib.load(path).get_fdata()
    img = _normalize_image(nifti, *args, **kwargs)
    if return_nifti:
        return img, nifti
    return img


class MissingGTError(Exception): ...


@lru_cache
def get_pid_to_index_lookup():
    stats = get_measure_statistics()
    ids = stats.dicom_metadata["DICOM;PatientID"]
    return {pid.split("_")[0]: i for i, pid in enumerate(ids)}


class MetadataSample:
    def __init__(self, path, target_shape, gt_type: MetadataTypes):
        """
        Creates a new ImageGT object from the given path.

        image has shape (18, 320, 896) and values in [0, 1].

        gt has shape (18, 320, 896) and values in {0, 1, 2}.
        gt_onehot has shape (3, 18, 320, 896) and values in {0, 1, 2}.

        If each_roi_is_separate_class is True, then the values will be in {0, 1, 2, ..., #num_rois},
        and the first dimension of gt_onehot will be #num_rois.
        """
        self.target_shape = target_shape

        self.pid = path.name
        self.image, self.nifti = _load_and_normalize(path / f"{self.pid}.nii.gz", target_shape, return_nifti=True)
        # self.wk = _load_and_normalize(path / f"{pid}_WK.nii.gz", is_gt=True)
        # self.bs = _load_and_normalize(path / f"{pid}_BS.nii.gz", is_gt=True)
        self.gt = self._compose_groundtruth(gt_type)

    def _compose_groundtruth(self, gt_types: MetadataTypes):
        import torch

        stats = get_measure_statistics()
        lookup = get_pid_to_index_lookup()
        try:
            row = stats.dicom_metadata.iloc[lookup[self.pid]]
        except KeyError:
            raise MissingGTError(f"Missing ground truth for this sample {self.pid=}")
        gt = []

        def handle_gt(gt_type):
            value = row[f"DICOM;Patient{gt_type.capitalize()}"]
            if gt_type == "sex":
                assert value in "MF"
                gt.append((value == "M"))
            elif gt_type == "age":
                assert value.endswith("Y")
                gt.append(int(value[:-1]) / 100)
            elif gt_type == "weight":
                gt.append(float(value) / 100)
            elif gt_type == "size":
                gt.append(float(value))
            else:
                raise ValueError(f"Unknown gt_type {gt_type=} {value=}")

        if isinstance(gt_types, str):
            gt_types = [gt_types]
        for type_ in sorted(set(gt_types)):
            handle_gt(type_)

        return torch.tensor(gt, dtype=torch.float32)


def get_random_crop_slice(crop_height_to_px, full_height, start_at=None):
    if start_at is None:
        crop_start_at = random.randint(0, full_height - crop_height_to_px)
    else:
        crop_start_at = start_at
    return slice(crop_start_at, crop_start_at + crop_height_to_px)


class Sample:
    def __init__(
        self,
        path,
        target_shape,
        gt_format: Optional[GTFormat],
        crop_height_to_px=None,
        zip_to_gt_path_lookup: Dict[Path, Path] = None,
    ):
        """
        Creates a new ImageGT object from the given path.

        image has shape (18, 320, 896) and values in [0.0, 1.0].
        gt has shape (18, 320, 896) and values in {0, 1, 2}.
        """
        self.target_shape = target_shape
        self.gt_format = gt_format
        self.path = Path(path).expanduser()
        self.dicom = None
        self.zip_to_gt_path_lookup = zip_to_gt_path_lookup

        if crop_height_to_px is None:
            crop_height_to_px = self.target_shape[2]
        self.crop_slice = get_random_crop_slice(crop_height_to_px, self.target_shape[2])

        init_methods = [self._init_if_nifti(self.path), self._init_if_zip(self.path)]
        assert sum(init_methods) == 1, f"Ensure exactly one loading init method was used, not {sum(init_methods)=})"

    def _init_if_zip(self, path) -> bool:
        if path.suffix != ".zip":
            return False

        series_id_to_dicoms: Dict[str, List[pydicom.FileDataset]] = defaultdict(list)
        with zipfile.ZipFile(path, "r") as zip_ref:
            for filename in zip_ref.namelist():
                with zip_ref.open(filename) as file:
                    dicom = pydicom.dcmread(BytesIO(file.read()))
                    series_id_to_dicoms[dicom.SeriesInstanceUID].append(dicom)

        images = []
        for series_id, dicoms in sorted(series_id_to_dicoms.items())[:1]:
            self.dicom = dicoms[0]
            for dicom in sorted(dicoms, key=lambda d: d.InstanceNumber):
                images.append(dicom.pixel_array.T[::-1, ::-1])

        self.image, self.full_image = _normalize_image(
            np.stack(images, axis=0), self.target_shape, return_full=True, crop_height_to_slice=self.crop_slice
        )
        if self.zip_to_gt_path_lookup:
            self.npz = np.load(self.zip_to_gt_path_lookup[path], allow_pickle=True)
            seg = self.npz["segmentation"]

            self.full_wk = (seg == 1).astype(int)
            self.wk = self.full_wk[:, :, self.crop_slice]

            self.full_bs = (seg == 2).astype(int)
            self.bs = self.full_bs[:, :, self.crop_slice]

            self.gt, self.full_gt = self._compose_groundtruth()
        else:
            self.full_gt = np.zeros_like(self.full_image)
            self.gt = self.full_gt[:, :, self.crop_slice]
        return True

    def _init_if_nifti(self, path) -> bool:
        pid = path.name

        image_path = path / f"{pid}.nii.gz"
        if not image_path.exists():
            return False

        self.image, self.full_image = _load_and_normalize(
            image_path, self.target_shape, return_full=True, crop_height_to_slice=self.crop_slice
        )
        if self.gt_format is not None:
            self.wk, self.full_wk = _load_and_normalize(
                path / f"{pid}_WK.nii.gz",
                self.target_shape,
                is_gt=True,
                return_full=True,
                crop_height_to_slice=self.crop_slice,
            )
            p = path / f"{pid}_BS.nii.gz"
            if p.exists():
                self.bs, self.full_bs = _load_and_normalize(
                    p, self.target_shape, is_gt=True, return_full=True, crop_height_to_slice=self.crop_slice
                )
            else:
                self.bs = self.full_bs = None
            self.gt, self.full_gt = self._compose_groundtruth()
        return True

    def _compose_groundtruth(self):
        compose_gt_func = getattr(self, f"_compose_gt_{self.gt_format.type}", None)
        if compose_gt_func is not None:
            return compose_gt_func()
        else:
            raise ValueError(f"Unknown gt_format.type '{self.gt_format.type=}'")

    def _compose_gt_classification(self):
        gtf = self.gt_format
        assert gtf.separation == "instance", f"{gtf.separation=}"
        assert gtf.order == "y-sort", f"{gtf.order=}"
        seg_gt, full_seg_gt = self._compose_gt_segmentation()
        # gt_length = 24 * ("bs" in gtf.include) + 25 * ("wk" in gtf.include)
        # print(seg_gt.shape)
        # print(seg_gt)
        uniq = np.unique(seg_gt)
        gt = np.array([n in uniq for n in range(1, 50)]).astype(float)
        # classes = [n for n in uniq if 1 <= n <= 49]
        # gt[classes] = 1
        ret = np.repeat(gt[None], self.target_shape[0], axis=0)
        return ret, ret

    def _compose_gt_segmentation(self):
        gt = np.zeros_like(self.full_wk)
        if self.full_bs is not None:
            wk, bs = self.full_wk, self.full_bs
        else:
            wk, bs = (self.full_wk % 2 == 0) & (self.full_wk > 0), self.full_wk % 2 == 1

        gtf = self.gt_format

        if gtf.separation == "instance":
            if hasattr(self, "npz") and "instances" in self.npz:
                instances = self.npz["instances"]
                id_to_label = self.npz["id_to_label"].item()
            else:
                wkbs = wk - ((wk != 0) & (bs != 0)) + bs * 2
                assert wkbs.shape[2] == 896, f"{wkbs.shape=}"
                instances, id_to_label = separate_rois_with_labels(wkbs)
            curr_index = 1
            for id_, label in sorted(id_to_label.items()):
                is_bs = "-" in label
                if not is_bs and "wk" in gtf.include:
                    if label == "S1" and gtf.s1_class is not None:
                        gt[instances == id_] = gtf.s1_class
                    else:
                        gt[instances == id_] = curr_index
                        curr_index += 1
                if is_bs and "bs" in gtf.include:
                    gt[instances == id_] = curr_index
                    curr_index += 1

        if gtf.separation == "semantic":
            if "wk" in gtf.include:
                gt[wk != 0] = 1
            if "bs" in gtf.include:
                gt[bs != 0] = gt.max() + 1

        if gtf.separation == "binary":
            if "wk" in gtf.include:
                gt[wk != 0] = 1
            if "bs" in gtf.include:
                gt[bs != 0] = 1

        return gt[:, :, self.crop_slice], gt

    def _onehot_encode(self):
        gt = self.gt
        gt_onehot = np.stack([(gt == value).astype(int) for value in sorted(np.unique(gt))])
        assert gt_onehot.min() == 0 and gt_onehot.max() == 1, f"{gt_onehot.min()=} {gt_onehot.max()=}"
        assert len(np.unique(gt_onehot)) == 2, f"{np.unique(gt_onehot)=}"
        return gt_onehot


@dataclass
class SampleIterator(Iterable[Tuple[np.ndarray, np.ndarray]]):
    data_dirs: List[Path]
    crop_height_to_px: int = None
    target_shape: Tuple[Optional[int]] = (None, 320, 896)
    add_adjacent_slices: bool = False
    already_processed: Set[Path] = field(default_factory=set)
    skip_first_percent: float = 0

    def __iter__(self):
        start_at = int(len(self.data_dirs) * self.skip_first_percent)
        for path in self.data_dirs[start_at:]:
            if path in self.already_processed:
                continue
            try:
                sample = Sample(path, self.target_shape, gt_format=GTFormat(), crop_height_to_px=self.crop_height_to_px)
                batch_img = []
                batch_gt = []
                for i in range(sample.image.shape[0]):
                    if self.add_adjacent_slices:
                        image, gt = sample.image[max(0, i - 1) : i + 2], sample.gt[i : i + 1]

                        if i == 0:
                            image = np.concatenate([image[:1], image], axis=0)
                        if i == sample.image.shape[0] - 1:
                            image = np.concatenate([image, image[-1:]], axis=0)

                        assert image.shape[0] == 3, f"{image.shape=}"

                    else:
                        image = sample.image[i : i + 1]
                        gt = sample.gt[i : i + 1]
                    batch_img.append(image)
                    batch_gt.append(gt)
                yield np.stack(batch_img, axis=0), np.stack(batch_gt, axis=0), sample, path
            except KeyboardInterrupt:
                raise
            except:
                print("Error")
                import traceback

                traceback.print_exc()


def main():
    gt_format = GTFormat()

    from dataclasses import fields
    from typing import Literal, get_args

    for field in fields(GTFormat):
        print(f"Field name: {field.name}")

        # If the field type is a Literal, iterate over its options
        if getattr(field.type, "__origin__", None) is Literal:
            literals = get_args(field.type)
            for literal in literals:
                print(f"Literal: {literal}")

        # Print the value of the field for the instance
        print(f"Value: {getattr(gt_format, field.name)}")


if __name__ == "__main__":
    main()

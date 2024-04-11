import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from spine_segmentation.datasets.augmentation import ImageAugmentation
from spine_segmentation.datasets.path_helper import expand_path_to_data_dirs
from spine_segmentation.datasets.sample import GTFormat, Sample
from spine_segmentation.resources.paths import NAKO_DATASET_PATH
from spine_segmentation.resources.other_paths import RAW_NAKO_DATASET_PATH


class SegmentationDataset(Dataset):
    def __init__(
        self,
        data_dirs: Iterable[Path],
        transform=None,
        *,
        gt_format: Optional[GTFormat] = GTFormat(),
        target_shape: Tuple[Optional[int], int, int] = (18, 320, 896),
        slice_wise: bool = True,
        add_bs_wk_as_channels: bool = False,
        add_adjacent_slices: bool = False,
        triple_same_slice: bool = False,
        crop_height_to_px: int = None,
        use_only_n_center_slices: Optional[int] = None,
        overwrite_length: int = None,
        index_to_zip_path: Dict[int, Path] = None,
        index_to_slice_index: Dict[int, int] = None,
        zip_to_gt_path_lookup: Dict[Path, Path] = None,
    ):
        super().__init__()
        self.target_shape = target_shape
        self.data_dirs = list(data_dirs)
        self.gt_format = gt_format
        self.slice_wise = slice_wise
        self.transform = transform
        self.add_bs_wk_as_channels = add_bs_wk_as_channels
        self.add_adjacent_slices = add_adjacent_slices
        self.triple_same_slice = triple_same_slice
        self.crop_height_to_px = crop_height_to_px or self.target_shape[2]
        self.use_only_n_center_slices = use_only_n_center_slices or self.target_shape[0]
        self.overwrite_length = overwrite_length
        self.zip_to_gt_path_lookup = zip_to_gt_path_lookup
        self.index_to_zip_path = index_to_zip_path
        self.index_to_slice_index = index_to_slice_index

        assert (
            add_adjacent_slices + triple_same_slice + add_bs_wk_as_channels <= 1
        ), "Only one of these can be true: add_adjacent_slices, triple_same_slice, add_bs_wk_as_channels"
        if self.use_only_n_center_slices is None:
            self.slice_count = None
        else:
            self.slice_count = (self.use_only_n_center_slices - self.add_adjacent_slices * 2) if self.slice_wise else 1

    def __len__(self):
        if self.overwrite_length is not None:
            return self.overwrite_length
        return len(self.data_dirs) * self.slice_count

    @lru_cache(maxsize=5)
    def _get_sample(self, path):
        return Sample(
            path,
            self.target_shape,
            gt_format=self.gt_format,
            crop_height_to_px=self.crop_height_to_px,
            zip_to_gt_path_lookup=self.zip_to_gt_path_lookup,
        )

    def __getitem__(self, index):
        if self.index_to_zip_path is not None:
            path = self.index_to_zip_path[index - self.index_to_slice_index[index]]
        else:
            path = self.data_dirs[index // self.slice_count]
        sample = self._get_sample(path)

        # print(index, sample, sample.image.shape, sample.gt.shape)

        if self.use_only_n_center_slices is None:
            start_at = 0
        else:
            start_at = self.target_shape[0] // 2 - self.use_only_n_center_slices // 2

        if self.slice_wise:
            if self.index_to_slice_index is not None:
                i = self.index_to_slice_index[index]
            else:
                i = start_at + index % self.slice_count
            # return sample.sample[i : i + 1], sample.gt_onehot[:, i : i + 1]
            if self.add_adjacent_slices:
                # image_slice = Tensor(sample.image[i : i + 3])
                curr_slice = Tensor(sample.image[i])
                valid_indices = range(sample.image.shape[0])
                image_slice = torch.stack(
                    [
                        Tensor(sample.image[i - 1]) if i - 1 in valid_indices else curr_slice,
                        curr_slice,
                        Tensor(sample.image[i + 1]) if i + 1 in valid_indices else curr_slice,
                    ],
                    dim=0,
                )
                gt_slice = Tensor(sample.gt[i : i + 1].astype(np.int16))
            else:
                image_slice, gt_slice = Tensor(sample.image[i : i + 1]), Tensor(sample.gt[i : i + 1].astype(np.int16))

            if self.add_bs_wk_as_channels:
                image_slice = torch.cat(
                    [
                        image_slice,
                        Tensor(sample.wk[i : i + 1]),
                        Tensor(sample.bs[i : i + 1]),
                    ],
                    dim=0,
                )
            if self.triple_same_slice:
                image_slice = torch.cat([image_slice, image_slice, image_slice], dim=0)
            if self.transform is not None:
                image_slice, gt_slice = self.transform.single_image(image_slice, gt_slice)
            # assert image_slice.shape == gt_slice.shape, f"{image_slice.shape=} != {gt_slice.shape=}"
            return image_slice, gt_slice
        else:
            assert not any((self.add_adjacent_slices, self.triple_same_slice, self.add_bs_wk_as_channels))
            return sample.image, sample.gt

    def plot(self, index=None):
        if index is None:
            index = np.random.randint(len(self))

        image, gt = self[index]
        image = image.numpy()
        gt = gt.numpy()
        if len(image.shape) == 4:
            image = image[9]
            gt = image[9]
        image_gt = image.copy()
        image_gt[gt != 0] = gt[gt != 0] / gt.max() * image.max()
        for i, img in enumerate([image, gt, image_gt]):
            plt.subplot(1, 3, i + 1)
            plt.imshow(img.T[::-1])
            plt.xticks([])
            plt.yticks([])
        shape = image.shape
        plt.tight_layout(pad=0.2)
        plt.title(f"Image ({index=}) ({shape=}) ")
        plt.show()

    def plot_augmentations(self, index=None, num_images=16, add_gt=False):
        if index is None:
            index = np.random.randint(len(self))

        for i in range(num_images):
            image, gt = self[index]
            image = image.numpy()
            if add_gt:
                image[gt != 0] = gt[gt != 0] / gt.max() * image.max()

            plt.subplot(2, num_images // 2, i + 1)
            plt.imshow(image.T[::-1])
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout(pad=0)
        # plt.title(f"Image ({index=}) ({image.shape=}) ")
        plt.gcf().set_dpi(300)
        plt.show()

    def plot_gt_variations(self, index=9):
        if index is None:
            index = np.random.randint(len(self))
        previous_gt_format = self.gt_format

        from dataclasses import fields
        from itertools import product
        from typing import get_args

        field_values = [get_args(field.type) for field in fields(GTFormat)]
        all_possible_args = list(product(*field_values))

        fig, axes = plt.subplots(2, len(all_possible_args) // 2, figsize=(10, 5))

        for i, (ax, values) in enumerate(zip(axes.flatten(), all_possible_args), 1):
            self.gt_format = GTFormat(*values)

            start_time = time.time()
            _, gt = self[index]
            print(f"{i}. values={values} elapsed_time={time.time() - start_time}")

            gt = gt.numpy()

            ax.imshow(gt.T[::-1])
            ax.set_title("+".join(values), fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout(pad=0)
        self.gt_format = previous_gt_format
        # plt.title(f"Image ({index=}) ({image.shape=}) ")
        plt.gcf().set_dpi(300)
        plt.show()


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path, List[str], Tuple[str]] = NAKO_DATASET_PATH,
        batch_size=1,
        num_workers=2,
        gt_format: GTFormat = GTFormat(),
        target_shape: Tuple[Optional[int], int, int] = (18, 320, 896),
        slice_wise=True,
        augment=False,
        add_bs_wk_as_channels=False,
        add_adjacent_slices=False,
        triple_same_slice=False,
        crop_height_to_px: int = None,
        use_only_n_center_slices: int = None,
        ignore_train_from_csv: List[Union[Path, str]] = None,
    ):
        super().__init__()
        self.batch_size = batch_size

        train_index_length = None
        val_index_length = None

        train_index_to_zip_path = None
        val_index_to_zip_path = None

        index_to_slice_index = None
        val_index_to_slice_index = None

        index_to_npz_path = None

        zip_path_to_npz_path = None

        if isinstance(data_dir, tuple) or isinstance(data_dir, list):
            train_path, val_path = data_dir
            train_paths = expand_path_to_data_dirs(train_path)
            val_paths = expand_path_to_data_dirs(val_path)
        else:
            self.data_dirs = expand_path_to_data_dirs(data_dir)

            assert self.data_dirs, f"No data found in {data_dir=}"

            train_size_percentage = 0.95
            train_size = int(len(self.data_dirs) * train_size_percentage)
            train_paths = self.data_dirs[:train_size]
            val_paths = self.data_dirs[train_size:]

            if Path(data_dir).suffix == ".npz":
                npz = np.load(data_dir, allow_pickle=True)
                index_to_zip_path = npz["index_to_zip_path"].item()
                index_to_slice_index = npz["index_to_slice_index"].item()
                index_to_npz_path = npz["index_to_npz_path"].item()

                common_indices = (
                    set(index_to_npz_path.keys()) & set(index_to_zip_path.keys()) & set(index_to_slice_index.keys())
                )
                index_to_zip_path = {index: index_to_zip_path[index] for index in common_indices}
                assert all(index in index_to_slice_index for index in common_indices)
                index_to_npz_path = {index: index_to_npz_path[index] for index in common_indices}
                assert all(index in index_to_zip_path for index in common_indices)
                assert all(index in index_to_npz_path for index in common_indices)

                first_val_index = next(index for index, path in index_to_zip_path.items() if path == val_paths[0])
                train_index_length = first_val_index - 1
                val_index_length = len(index_to_slice_index) - train_index_length
                assert val_index_length + train_index_length == max(index_to_slice_index) + 1
                val_index_length -= 21

                train_index_to_zip_path = {
                    index: path for index, path in index_to_zip_path.items() if index < first_val_index
                }
                val_index_to_zip_path = {
                    index - first_val_index: path
                    for index, path in index_to_zip_path.items()
                    if index >= first_val_index
                }

                val_index_to_slice_index = {
                    index - first_val_index: slice_index
                    for index, slice_index in index_to_slice_index.items()
                    if index >= first_val_index
                }

                common_indices = set(index_to_npz_path.keys()) & set(index_to_zip_path.keys())
                zip_path_to_npz_path = {index_to_zip_path[index]: index_to_npz_path[index] for index in common_indices}

        if ignore_train_from_csv is not None:
            for ignore_path in ignore_train_from_csv:
                ignore_csv = pd.read_csv(ignore_path)
                ignore_paths = ignore_csv["Path"]

                print("============== Ignore ==============")
                print(f"{len(train_paths)=}")
                print(f"{len(ignore_paths)=}")
                ignore_paths = {Path(path).name for path in ignore_paths}
                print(list(ignore_paths)[:10])
                print("============")
                print(train_paths[:10])
                print("============")
                train_paths = [
                    p for p in train_paths if p.name not in ignore_paths and p.name.split("_")[0] not in ignore_paths
                ]
                print(f"{len(train_paths)=}")

        self.segmentation_dataloader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
        )

        segmentation_dataset_kwargs = dict(
            gt_format=gt_format,
            slice_wise=slice_wise,
            add_bs_wk_as_channels=add_bs_wk_as_channels,
            add_adjacent_slices=add_adjacent_slices,
            triple_same_slice=triple_same_slice,
            crop_height_to_px=crop_height_to_px,
            use_only_n_center_slices=use_only_n_center_slices,
            zip_to_gt_path_lookup=zip_path_to_npz_path,
            target_shape=target_shape,
        )

        train_args = dict(
            transform=ImageAugmentation() if augment else None,
            overwrite_length=train_index_length,
            index_to_zip_path=train_index_to_zip_path,
            index_to_slice_index=index_to_slice_index,
        )

        val_args = dict(
            overwrite_length=val_index_length,
            index_to_zip_path=val_index_to_zip_path,
            index_to_slice_index=val_index_to_slice_index,
        )

        self.train_dataset = SegmentationDataset(train_paths, **train_args, **segmentation_dataset_kwargs)
        self.val_dataset = SegmentationDataset(val_paths, **val_args, **segmentation_dataset_kwargs)
        from spine_segmentation.utils.globals import TRAIN_DATASET_SIZE

        TRAIN_DATASET_SIZE.set(len(self.train_dataloader()))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.segmentation_dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.segmentation_dataloader_kwargs)


if __name__ == "__main__":
    # noinspection PyUnreachableCode
    if False:
        dataset = SegmentationDataset(get_potential_data_dirs(NAKO_DATASET_PATH))
        dataset.plot()

    # noinspection PyUnreachableCode
    if False:
        dataset = SegmentationDataset(get_potential_data_dirs(NAKO_DATASET_PATH), ImageAugmentation())
        dataset.plot_augmentations(add_gt=True)

    # noinspection PyUnreachableCode
    if False:
        dataset = SegmentationDataset(get_potential_data_dirs(NAKO_DATASET_PATH))
        dataset.plot_gt_variations()

    # noinspection PyUnreachableCode
    if True:
        dataset = SegmentationDataset(list(RAW_NAKO_DATASET_PATH.glob("*.zip"))[:100], gt_format=None)
        print(dataset[0])

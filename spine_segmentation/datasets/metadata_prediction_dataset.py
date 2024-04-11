import time
from pathlib import Path
from typing import Iterable

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from spine_segmentation.datasets.sample import GTFormat, MetadataSample, MetadataTypes, get_pid_to_index_lookup
from spine_segmentation.datasets.path_helper import get_potential_data_dirs
from spine_segmentation.resources.paths import NAKO_DATASET_PATH


class MetadataPredictionDataset(Dataset):
    def __init__(self, data_dirs: Iterable[Path], transform=None, gt_type: MetadataTypes = ["weight", "size", "age"]):
        super().__init__()
        self.target_shape = (18, 320, 896)
        self.data_dirs = list(data_dirs)
        self.transform = transform
        self.gt_type = gt_type
        self.add_adjacent_slices = True
        self.slice_wise = True
        self.slice_count = (self.target_shape[0] - self.add_adjacent_slices * 6) if self.slice_wise else 1

    def __len__(self):
        return len(self.data_dirs) * self.slice_count

    def __getitem__(self, index):
        path = self.data_dirs[index // self.slice_count]
        image = MetadataSample(path, self.target_shape, gt_type=self.gt_type)
        i = index % self.slice_count + 3
        img_slice = Tensor(image.image[i])
        image_slice = torch.stack([img_slice] * 3)
        gt = image.gt
        if self.transform is not None:
            image_slice = self.transform.single_image(image_slice)
        return image_slice, gt

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


class MetadataPredictionModule(pl.LightningDataModule):
    def __init__(self, data_dir=NAKO_DATASET_PATH, batch_size=1, num_workers=2, gt_type: MetadataTypes = "weight"):
        super().__init__()
        self.batch_size = batch_size
        gt_available = set(get_pid_to_index_lookup().keys())
        self.data_dirs = get_potential_data_dirs(data_dir, required_suffixes=[""], intersection_with=gt_available)
        train_size = int(len(self.data_dirs) * 0.9)
        train_paths = self.data_dirs[:train_size]
        val_paths = self.data_dirs[train_size:]

        self.dataloader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
        )

        dataset_kwargs = dict(
            gt_type=gt_type,
        )

        self.train_dataset = MetadataPredictionDataset(train_paths, **dataset_kwargs)
        self.val_dataset = MetadataPredictionDataset(val_paths, **dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.dataloader_kwargs)


def mean_prediction():
    # datamodule = MetadataPredictionModule()
    # train = datamodule.train_dataset
    # gts = np.array([gt for _, gt in train])
    # val = datamodule.val_dataset

    from spine_segmentation.resources.preloaded import get_measure_statistics

    stats = get_measure_statistics()

    gt_types = ["age", "sex", "weight", "size"]
    gts = []
    for i in range(len(stats.dicom_metadata)):
        row = stats.dicom_metadata.iloc[i]
        if i % 100 == 0:
            print(i)
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
        gts.append(gt)
    arr = np.array(gts)
    mean = arr.mean(axis=0)
    mse = ((arr - mean) ** 2).mean(axis=0)
    msvariance = mse**0.5
    print(arr.shape)


if __name__ == "__main__":
    # noinspection PyUnreachableCode
    if True:
        mean_prediction()

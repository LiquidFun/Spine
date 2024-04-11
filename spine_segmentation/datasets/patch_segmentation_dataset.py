from pathlib import Path
from typing import Iterable, Literal

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spine_segmentation.datasets.augmentation import ImageAugmentation
from spine_segmentation.datasets.sample import GTFormat, Sample
from spine_segmentation.datasets.path_helper import get_potential_data_dirs
from spine_segmentation.resources.paths import NAKO_DATASET_PATH


class PatchSegmentationDataset(Dataset):
    def __init__(
        self, data_dirs: Iterable[Path], transform=None, subset_size: int = 20, mode: Literal["val", "train"] = "val"
    ):
        super().__init__()
        self.target_shape = (18, 320, 896)
        self.patch_target_shape = (18, 128, 128)
        self.data_dirs = list(data_dirs)
        self.gt_format = GTFormat(order="y-sort", separation="instance", include="wk")
        self.transform = transform
        self.mode = mode
        self.subset_size = subset_size

    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, index):
        path = self.data_dirs[index]
        image = Sample(path, self.target_shape, gt_format=self.gt_format)
        # connected_components = numerate_objects(image.gt)
        outputs = []
        targets = []
        # while True:
        # start_at, end_at = sorted(np.random.randint(1, max(np.unique(image.gt)), size=(2,)))
        # if end_at - start_at >= 5:
        #     break

        start_at = np.random.randint(1, max(np.unique(image.gt)) - self.subset_size)
        end_at = start_at + self.subset_size
        assert end_at - start_at == self.subset_size
        assert end_at <= max(np.unique(image.gt))

        for val in np.unique(image.gt)[:50]:
            if val == 0:
                continue
            if not (start_at <= val <= end_at):
                continue
            # Average coordinate of the object
            coords = np.argwhere(image.gt == val)
            coords = np.round(coords.mean(axis=0)).astype(int)

            if self.mode == "train":
                coords += np.random.randint(-5, 5, coords.shape)

            shape = self.patch_target_shape

            # Sample self.patch_target_shape around the coordinate
            patch = image.image[
                :,  # coords[0] - shape[0] // 2 : coords[0] + shape[0] // 2,
                coords[1] - shape[1] // 2 : coords[1] + shape[1] // 2,
                coords[2] - shape[2] // 2 : coords[2] + shape[2] // 2,
            ]

            # Pad if necessary
            padding = np.subtract(shape, patch.shape)
            padding[padding < 0] = 0
            padding = padding // 2
            padded_patch = np.pad(patch, ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2])))
            padded_patch = torch.from_numpy(padded_patch)

            gt_patch = image.gt[
                :,  # coords[0] - shape[0] // 2 : coords[0] + shape[0] // 2,
                coords[1] - shape[1] // 2 : coords[1] + shape[1] // 2,
                coords[2] - shape[2] // 2 : coords[2] + shape[2] // 2,
            ]

            padded_gt_patch = np.pad(
                gt_patch, ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2]))
            )
            padded_gt_patch = (padded_gt_patch == val).astype(np.float32)
            padded_gt_patch = torch.from_numpy(padded_gt_patch)

            focus_patch = padded_patch * (padded_gt_patch + 0.5)

            if _plot_patch := False:
                from matplotlib import pyplot as plt

                plt.subplot(2, 2, 1)
                current_gt = (image.gt == val).astype(float)
                current_gt[
                    :,  # coords[0] - shape[0] // 2 : coords[0] + shape[0] // 2,
                    coords[1] - shape[1] // 2 : coords[1] + shape[1] // 2,
                    coords[2] - shape[2] // 2 : coords[2] + shape[2] // 2,
                ] += 0.5

                plt.imshow(current_gt[9])
                print(coords)

                plt.subplot(2, 2, 2)
                print(patch.shape)
                plt.imshow(patch[9])

                gt_patch = image.gt[
                    :,  # coords[0] - shape[0] // 2 : coords[0] + shape[0] // 2,
                    coords[1] - shape[1] // 2 : coords[1] + shape[1] // 2,
                    coords[2] - shape[2] // 2 : coords[2] + shape[2] // 2,
                ]

                plt.subplot(2, 2, 3)
                plt.imshow(gt_patch[9])

                plt.subplot(2, 2, 4)
                plt.imshow(focus_patch[9])
                plt.suptitle(f"Ind: {val}")
                plt.show()
                print("Shown\n")

            # outputs.append(padded_patch[None, ...])
            # outputs.append(torch.stack([padded_patch, padded_gt_patch]))
            outputs.append(focus_patch[None, ...])
            targets.append(val.astype(np.float32))
            # targets.append(val.astype(int))

        return outputs, torch.tensor(targets)


class PatchSegmentationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir=NAKO_DATASET_PATH, batch_size=1, num_workers=2, subset_size: int = 20, augment=False):
        super().__init__()
        self.batch_size = batch_size
        self.data_dirs = get_potential_data_dirs(data_dir)
        train_size = int(len(self.data_dirs) * 0.9)
        train_paths = self.data_dirs[:train_size]
        val_paths = self.data_dirs[train_size:]

        self.segmentation_dataloader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
        )

        segmentation_dataset_kwargs = dict()

        augmentation = ImageAugmentation() if augment else None
        self.train_dataset = PatchSegmentationDataset(
            train_paths, augmentation, mode="train", subset_size=subset_size, **segmentation_dataset_kwargs
        )
        self.val_dataset = PatchSegmentationDataset(val_paths, subset_size=subset_size, **segmentation_dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.segmentation_dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.segmentation_dataloader_kwargs)


if __name__ == "__main__":
    dataset = PatchSegmentationDataset(get_potential_data_dirs(NAKO_DATASET_PATH))
    dataset[0]

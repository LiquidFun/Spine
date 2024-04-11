import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset

from spine_segmentation.resources.preloaded import get_measure_statistics


class FeatureDataset(Dataset):
    def __init__(self, vector_table, size=10):
        patients, rois, axes = vector_table.shape
        # vector_table = np.nan_to_num(vector_table)
        self.num_classes = diff = rois - size
        samples = patients * diff
        vector_table = np.nan_to_num(vector_table)

        self.data = np.zeros((samples, size, axes)).astype(np.float32)
        self.labels = np.zeros((samples,)).astype(int)
        for start in range(0, diff):
            sample = vector_table[:, start : start + size, :]
            # print(np.isnan(sample).sum())
            # if not np.isnan(sample).any():
            self.data[start::diff, :, :] = sample
            self.labels[start::diff] = start + 1

        # self.data = np.zeros((samples, size, axes)).astype(np.float32)
        # self.labels = np.zeros((samples,)).astype(int)
        # start_indices = np.arange(diff)
        # self.data[start_indices[:, None] * diff + np.arange(samples // diff)[:, None, None], :, :] \
        #     = vector_table[:, start_indices[:, None] + np.arange(size)[None, :, None], :]
        # self.labels[start_indices[:, None] * diff + np.arange(samples // diff)[:, None]] = start_indices[:, None]

        # shape: (10425*37, 10, 3)
        # self.data = np.zeros((samples, size, axes)).astype(np.float32)
        # self.labels = np.zeros((samples,)).astype(int)

        # # create an array of start indices [[0], [1], ..., [diff-1]], shape: (diff, 1)
        # start_indices = np.arange(diff)[:, None]

        # # Compute the indices in self.data and self.labels where the values will be assigned.
        # into_indices = start_indices * diff + np.arange(samples // diff)[:, None, None]
        # from_indices = start_indices + np.arange(size)[None, :, None]

        # # Assign the values from vector_table to self.data
        # self.data[into_indices, :, :] = vector_table[:, from_indices, :]

        # # Assign the values from start_indices to self.labels
        # self.labels[into_indices.squeeze()] = start_indices

        # assert np.all(self.data_slow == self.data)
        # assert np.all(self.labels_slow == self.labels)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return self.data.shape[0]


class FeatureDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir=None, batch_size: int = 1, num_workers: int = 2, sample_size=10, train_split: float = 0.9
    ):
        super().__init__()
        stats = get_measure_statistics()
        only_for_vertebra = ["ANGLE_COR", "ANGLE_SAG", "HEIGHT_MIN"]
        too_correlated = ["_CENTER"]
        exclude = only_for_vertebra + too_correlated
        # self.features = stats.filter_by_column(
        #     required_substrings=["DIRECTION"], excluded_substrings=["CANAL_DIRECTION"], reshape=True
        # )

        self.features = stats.filter_by_column(
            required_substrings=["DIRECTION", "HEIGHT_MAX", "VOLUME", "LENGTH_AP"],
            excluded_substrings=["CANAL_DIRECTION", "VOLUME_PROB", "VOLUME_DIFF"],
            reshape=True,
        )
        # self.features = stats.filter_by_column(excluded_substrings=exclude, reshape=True)
        self.batch_size = batch_size
        train_size = round(self.features.shape[0] * train_split)
        train_features = self.features[:train_size]
        val_features = self.features[train_size:]

        self.dataloader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
        )

        dataset_kwargs = dict(
            size=sample_size,
        )

        self.train_dataset = FeatureDataset(train_features, **dataset_kwargs)
        self.val_dataset = FeatureDataset(val_features, **dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.dataloader_kwargs)


if __name__ == "__main__":
    stats = get_measure_statistics()
    FeatureDataset(stats.canal_direction_vectors)

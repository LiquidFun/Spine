import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from spine_segmentation.resources.preloaded import get_measure_statistics


class VectorTableGraphs(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, device="cpu"):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], map_location=device)

    @property
    def raw_file_names(self):
        return [""]

    @property
    def processed_file_names(self):
        return ["graphs_all.pt"]

    def download(self):
        pass

    def get_graphs(self, vector_table, size=10):
        patients, rois, axes = vector_table.shape
        vector_table = np.nan_to_num(vector_table)
        num_classes = diff = rois - size
        graphs = []
        r = np.arange(size - 1)
        edges = torch.tensor([r, r + 1], dtype=torch.long)
        for g in range(patients):
            # If mod 100
            if g % 100 == 0:
                print(g)
            for start in range(diff):
                features = vector_table[g, start : start + size, :]
                label = torch.tensor([start])
                graph = Data(x=features, edge_index=edges, y=label)
                graphs.append(graph)
        return graphs

    def process(self):
        # Read data into huge `Data` list.
        vector_table = get_measure_statistics().filter_by_column(
            excluded_substrings=["_PROB", "_DIFF", "ANGLE_COR", "ANGLE_SAG", "HEIGHT_MIN"],
            reshape=True,
        )
        data_list = self.get_graphs(vector_table)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        data.x = torch.Tensor(data.x)
        torch.save((data, slices), self.processed_paths[0])

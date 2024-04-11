from pathlib import Path
from typing import Union, List

import numpy as np

from spine_segmentation.resources.other_paths import RAW_NAKO_DATASET_PATH
from spine_segmentation.resources.paths import NAKO_DATASET_PATH


def expand_path_to_data_dirs(path: Union[Path, str]) -> List[Path]:
    path = Path(path)
    if path.suffix == ".npz":
        npz = np.load(path, allow_pickle=True)
        return list(npz["index_to_zip_path"].item().values())

    if path.suffix == ".csv":
        import pandas as pd

        csv = pd.read_csv(path)
        return list(csv["Path"])

    if path == RAW_NAKO_DATASET_PATH:
        return list(RAW_NAKO_DATASET_PATH.glob("*.zip"))

    if path == NAKO_DATASET_PATH:
        return get_potential_data_dirs(path, required_suffixes=("", "_WK"))

    raise ValueError(f"Invalid path {path=}")


def get_potential_data_dirs(data_dir, required_suffixes=("", "_BS", "_WK", "_NR"), intersection_with=None):
    if intersection_with is None:

        class AlwaysTrue:
            def __contains__(self, _):
                return True

        intersection_with = AlwaysTrue()
    paths = []
    for path in Path(data_dir).glob("*"):
        if path.name in intersection_with:
            if all((path / f"{path.name}{suffix}.nii.gz").exists() for suffix in required_suffixes):
                paths.append(path)
    return paths

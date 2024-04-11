import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from spine_segmentation.resources.paths import DATASETS_PATH, NAKO_DATASET_PATH

SPLITS_PATH = DATASETS_PATH / "nako_splits"
SPLITS_PATH.mkdir(parents=True, exist_ok=True)
TRAIN_SEG_PATH = SPLITS_PATH / "train_seg.csv"
VAL_SEG_PATH = SPLITS_PATH / "val_seg.csv"
TRAIN_INSTSEG_PATH = SPLITS_PATH / "train_instseg.csv"
VAL_INSTSEG_PATH = SPLITS_PATH / "val_instseg.csv"


TEST_PATIENT_ID_FILE_PATH = Path(__file__).parent / "test_patient_ids.txt"

INDEX_LIST_PATH = Path.home() / "<define-this-path>"


def save_as_csv(pat_id_to_path: Dict, path):
    print(f"Saving {len(pat_id_to_path)} to {path}")
    pat_id_to_path = OrderedDict(sorted(pat_id_to_path.items()))
    df = pd.DataFrame.from_dict(pat_id_to_path, orient="index", columns=["Path"])
    df["Path"] = df["Path"].astype(str)
    prefixes = [
        "",
    ]
    for prefix in prefixes:
        assert prefix[-1] != "/", f"Prefix {prefix} should not end with '/'"
        df["Path"] = df["Path"].str.replace(prefix, "~", regex=False)
    df.to_csv(path)


def main():
    full_dataset = {}
    full_gt_dataset = {}
    gt_only = {}

    test_set_ids = TEST_PATIENT_ID_FILE_PATH.open().read().strip().split("\n")
    print(test_set_ids)

    trainval_raw_dataset = {}
    trainval_truthed_dataset = {}

    test_set = {}

    # for path in RAW_NAKO_DATASET_PATH.glob("*"):
    #     pat_id = path.name.replace("_30_Sag T2 Spine.zip", "")
    #     full_dataset[pat_id] = path

    #     if pat_id not in test_set_ids:
    #         trainval_raw_dataset[pat_id] = path

    npz = np.load(INDEX_LIST_PATH, allow_pickle=True)

    for index, path in npz["index_to_zip_path"].item().items():
        pat_id = re.sub(r"_\d[05]_sag t2 spine.zip", "", path.name.lower())
        assert re.fullmatch(r"1\d{5}", pat_id), f"Invalid patient id {pat_id}"
        full_dataset[pat_id] = path

        if pat_id not in test_set_ids:
            trainval_raw_dataset[pat_id] = path

    for path in NAKO_DATASET_PATH.glob("*"):

        def get_gt_type(p):
            return p.name.replace(".", "_").split("_")[1]

        gt_types = [get_gt_type(p) for p in path.glob("*_*.nii.gz")]
        pat_id = path.name
        if "WK" in gt_types and "BS" in gt_types:
            if path.name not in full_dataset:
                gt_only[pat_id] = path
            full_gt_dataset[pat_id] = path
            full_dataset[pat_id] = path

            if pat_id not in test_set_ids:
                trainval_truthed_dataset[pat_id] = path
            else:
                test_set[pat_id] = path
        else:
            if pat_id in test_set_ids:
                print("testset patid no WK or BS", pat_id)

    assert all(pid in full_gt_dataset for pid in test_set_ids), "Not all test set ids not in gt_dataset"

    # paths.append(path.name)
    print(len(gt_only))
    print(*gt_only)

    seg_train_set = trainval_truthed_dataset
    seg_val_set = test_set

    import random

    random.seed(42)
    train_percentage = 0.95
    train_size = int(len(trainval_raw_dataset) * train_percentage)
    trainval_pat_ids = list(trainval_raw_dataset.keys())
    random.shuffle(trainval_pat_ids)
    train_ids, val_ids = trainval_pat_ids[:train_size], trainval_pat_ids[train_size:]

    union = set(seg_train_set) & set(val_ids)
    val_ids = set(val_ids) - union
    train_ids = set(train_ids) | union

    instseg_train_set = {pid: trainval_raw_dataset[pid] for pid in train_ids}
    instseg_val_set = {pid: trainval_raw_dataset[pid] for pid in val_ids}
    print(len(instseg_train_set), len(instseg_val_set))
    print(list(instseg_val_set)[:10])

    assert seg_val_set.keys() & instseg_train_set.keys() == set(), "Train and val sets should be disjoint"
    assert seg_val_set.keys() & instseg_val_set.keys() == set(), "Train and val sets should be disjoint"
    assert seg_val_set.keys() & seg_train_set.keys() == set(), "Train and val sets should be disjoint"

    assert instseg_val_set.keys() & instseg_train_set.keys() == set(), "Train and val sets should be disjoint"
    assert (
        instseg_val_set.keys() & seg_train_set.keys() == set()
    ), f"Should be disjoint, but got {instseg_val_set.keys() & seg_train_set.keys()}"

    save_as_csv(seg_train_set, TRAIN_SEG_PATH)
    save_as_csv(seg_val_set, VAL_SEG_PATH)
    save_as_csv(instseg_train_set, TRAIN_INSTSEG_PATH)
    save_as_csv(instseg_val_set, VAL_INSTSEG_PATH)


if __name__ == "__main__":
    main()

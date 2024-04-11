from functools import partial
from multiprocessing import Manager, Pool, cpu_count

import numpy as np
from tqdm import tqdm

from spine_segmentation.cli.cli import colored_tracebacks
from spine_segmentation.inference.onnx_model import ONNXInferenceModel

colored_tracebacks()

common_cases = [
    "C2:C2-C3:C3:C3-C4:C4:C4-C5:C5:C5-C6:C6:C6-C7:C7:C7-T1:T1:T1-T2:T2:T2-T3:T3:T3-T4:T4:T4-T5:T5:T5-T6:T6:T6-T7:T7"
    ":T7-T8:T8:T8-T9:T9:T9-T10:T10:T10-T11:T11:T11-T12:T12:T12-L1:L1:L1-L2:L2:L2-L3:L3:L3-L4:L4:L4-L5:L5:L5-S1:S1",
    "",
    "C2:C2-C3:C3:C3-C4:C4:C4-C5:C5:C5-C6:C6:C6-C7:C7:C7-T1:T1:T1-T2:T2:T2-T3:T3:T3-T4:T4:T4-T5:T5:T5-T6:T6:T6-T7:T7"
    ":T7-T8:T8:T8-T9:T9:T9-T10:T10:T10-T11:T11:T11-T12:T12:T12-L1:L1:L1-L2:L2:L2-L3:L3:L3-L4:L4:L4-L5:L5:L5-L6:L6:L6"
    "-S1:S1",
    "",
    "C2:C2-C3:C3:C3-C4:C4:C4-C5:C5:C5-C6:C6:C6-C7:C7:C7-T1:T1:T1-T2:T2:T2-T3:T3:T3-T4:T4:T4-T5:T5:T5-T6:T6:T6-T7:T7"
    ":T7-T8:T8:T8-T9:T9:T9-T10:T10:T10-T11:T11:T11-T12:T12:T12-L1:L1:L1-L2:L2:L2-L3:L3:L3-L4:L4:L4-S1:S1",
]


def process_path(path, counter, lock, rare_cases):
    npz = np.load(path, allow_pickle=True)
    id_to_label = npz["id_to_label"].item()
    key = ":".join(id_to_label.values())
    with lock:
        if key not in counter:
            counter[key] = 0
        counter[key] += 1
        if key not in common_cases:
            print(f"Rare case: {path}")
            rare_cases.append(path)


def main():
    onnx = ONNXInferenceModel.get_best_segmentation_model()
    onnx.load_index_list()
    # Tqdm loading and multiprocessing:
    # path = (Path.home() / "devel/src/git" / path.relative_to(relative_to)).expanduser()

    # Assuming onnx.index_to_npz_path.values() is a list of paths
    paths = list(onnx.index_to_npz_path.values())

    with Manager() as manager:
        counter = manager.dict()  # shared counter among processes
        rare_cases = manager.list()
        counter_lock = manager.Lock()  # lock to safely update counter

        func = partial(process_path, counter=counter, lock=counter_lock, rare_cases=rare_cases)

        with Pool(cpu_count()) as pool:
            for _ in tqdm(pool.imap_unordered(func, paths), total=len(paths)):
                pass

        # Convert manager counter to a normal dictionary
        counter_dict = dict(counter)
        rare_cases = list(rare_cases)

    for key, value in counter_dict.items():
        print(f"{value}: len={len(key)} {key}")

    for case in rare_cases:
        print(case)


if __name__ == "__main__":
    main()

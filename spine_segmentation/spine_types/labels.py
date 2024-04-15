from itertools import zip_longest
from typing import List, Dict


def get_vertebra_labels(c_range=range(2, 8), t_range=range(1, 13), l_range=range(1, 6)):
    return [*[f"C{i}" for i in c_range], *[f"T{i}" for i in t_range], *[f"L{i}" for i in l_range], "S1"]


def get_disc_labels(vertebra_labels):
    return [
        *[f"{upper}-{lower}" for upper, lower in zip(vertebra_labels[:-1], vertebra_labels[1:])],
    ]


# 6 + 12 + 5 + 1
def get_labels_for_n_classes(n=47) -> List[str]:
    while n not in [0, 45, 47, 49] and n > 45:
        print(f"Only 45, 47, 49 classes are supported, got {n}, downscaling for now")
        n -= 1

    # assert n in [45, 47, 49], "Only 45, 47, 49 classes are supported"
    lumbar_vertebra_count = 4 + (n >= 47) + (n >= 49)
    lumbar_range = range(1, lumbar_vertebra_count + 1)
    vertebra_labels = get_vertebra_labels(c_range=range(2, 8), t_range=range(1, 13), l_range=lumbar_range)
    disc_labels = get_disc_labels(vertebra_labels)

    labels = list(filter(None, sum(zip_longest(vertebra_labels, disc_labels), ())))
    return labels


def get_label_lookup_for_n_classes(n=47) -> Dict[int, str]:
    labels = get_labels_for_n_classes(n=n)
    label_lookup = {0: "0_unknown"}
    for i, label in enumerate(labels, 1):
        label_lookup[label] = i
    return label_lookup
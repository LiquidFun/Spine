from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from spine_segmentation.resources.paths import LOGS_PATH
from spine_segmentation.spine_types.labels import get_labels_for_n_classes
from spine_segmentation.visualisation.color_palettes import get_wk_bs_palette


def plot_accuracy(correct_counter: Counter[int, int], appeared_counter: Counter[int, int], V: int, N: int):
    # Calculate accuracy for each class

    matplotlib.rc("font", size=21)

    labels = get_labels_for_n_classes(49)
    # to_x = lambda x: x + (x > 12) * 1 + (x > 36) * 1
    to_x = lambda x: x  # + (x > 12) * 1 + (x > 36) * 1

    for i in set(correct_counter):
        if i not in range(1, 50):
            del correct_counter[i]
    for i in set(appeared_counter):
        if i not in range(1, 50):
            del appeared_counter[i]

    accuracy = {to_x(k): correct_counter[k] / appeared_counter[k] for k in sorted(appeared_counter.keys())}

    # Prepare data for seaborn
    data = pd.DataFrame(list(accuracy.items()), columns=["Class", "Accuracy"])  # , index=map(to_x, range(49)))

    palette = get_wk_bs_palette()
    del palette[0]
    colors = np.array([list(c.floats()) for c in palette.values()])
    colors = np.tile(colors.T, 100).T[:49]
    is_thoracic = np.array([l.startswith("T") for l in labels])
    # colors[is_thoracic] *= 0.7
    # data["Hue"] = [v.ints() for k, v in palette.items()]
    # palette = {int(k): v.ints() for k, v in palette.items()}
    # Create seaborn barplot
    plt.figure(figsize=(10, 6))
    # sns.barplot(x="Class", y="Accuracy", dodge=False, data=data, palette=colors)
    bars = plt.bar(data["Class"], data["Accuracy"], color=colors)
    plt.margins(x=0.003)
    plt.legend(bars[:2], ["Vertebra", "Intervertebral disc"], loc="lower left")
    # plt.bar(range(50), accuracy.values())
    # plt.title(f"Multiclass Segmentation Accuracy per Vertebra and Intervertebral Disc for FOV: V={V} and N={N}")
    # print(len(labels), len(range()))
    # fontsizes = [20 if i % 2 == 1 else 10 for i in range(49)]
    vertebra_labels = [label if i % 2 == 0 else "" for i, label in enumerate(labels)]
    plt.xticks(list(map(to_x, range(1, 50))), vertebra_labels, rotation=90)
    plt.ylabel("Accuracy")

    yticks = np.linspace(0, 1, 21)
    percent = [f"{int(y * 100)}%" for y in yticks]
    plt.yticks(yticks, percent)
    ax = plt.gca()  # Get current axes

    for i, line in enumerate(ax.get_ygridlines()):
        if i % 2 == 0:  # for even indices
            line.set(alpha=0.8)
            line.set_linestyle("-")
            line.set_linewidth(1)
        else:  # for odd indices
            line.set_linestyle("-")
            line.set_linewidth(1)
            line.set(alpha=0.3)

    # plt.grid(axis="y", color=[0, 0, 0], alpha=0.2)
    plt.grid(axis="y", color=[0, 0, 0])
    plt.tight_layout(pad=0)
    plot_path = Path(__file__).parent / "plots" / "accuracy_per_class" / f"PerROIAccuracyV{V}.png"
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.savefig(f"/tmp/perroiaccuracy/{plot_path.name}", dpi=300)
    plt.show()


def main():
    # log_dirs = ["2023-11-06_232432", "2023-11-06_232433", "2023-11-06_232435"]
    # log_dirs = ["2023-11-06_232433"]

    # 30k but bad appeared counter
    # log_dirs = [
    #     ("2023-11-07_210729", 15),
    #     ("2023-11-07_210805", 10),
    #     ("2023-11-07_210824", 5),
    # ]

    # Good nice data
    log_dirs = [
        ("2023-11-08_134517", 20),
        ("2023-11-08_134323", 15),
        ("2023-11-08_134534", 10),
        ("2023-11-08_134551", 5),
    ]

    # Something fixed here ?
    # log_dirs = [
    #     ("2023-11-08_185546", 15),
    #     ("2023-11-08_185614", 10),
    #     ("2023-11-08_185820", 5),
    # ]

    # Test set with new segmentation and new inst segmentation
    log_dirs = [
        ("2023-11-09_044759", 15),
        ("2023-11-09_044842", 10),
        ("2023-11-09_044825", 5),
    ]

    log_dirs = [("2023-11-09_012718", 8)]

    log_dirs = [
        ("2023-11-09_144956", 5),
        ("2023-11-09_144959", 7),
        ("2023-11-09_145015", 10),
        ("2023-11-09_145032", 15),
    ]

    for log_dir_name, V in log_dirs:
        log_path = LOGS_PATH / log_dir_name
        npz_path = log_path / "stats.npz"
        npz = np.load(npz_path, allow_pickle=True)

        stats = npz["stats"]

        correct_counter = Counter()
        correct_no_edges_counter = Counter()
        appeared_counter = Counter()
        appeared_no_edges_counter = Counter()
        dice_scores = []

        corr = []
        corr_no_edges = []

        print(len(stats))
        for pat in stats:
            if "correct_counter" in pat and "correct_no_edges_counter" in pat:
                correct_counter += pat["correct_counter"]
                correct_no_edges_counter += pat["correct_no_edges_counter"]
                appeared_counter += pat["appeared_counter"]
                appeared_no_edges_counter += pat["appeared_no_edges_counter"]
            if "f1" in pat:
                dice_scores.append(pat["f1"])
                corr_no_edges.append(pat["correct_no_edges"])
                corr.append(pat["correct"])
        del appeared_counter[0]
        plot_accuracy(correct_no_edges_counter, appeared_no_edges_counter, V=V, N=sum(appeared_counter.values()))

        dice_scores = np.array(dice_scores)

        # print(correct_counter)
        print(sorted(correct_no_edges_counter.items()))
        print(sorted(appeared_counter.items()))
        print(f"Dice: (V={V})", np.mean(dice_scores))
        print(f"Correct: (V={V})", np.mean(corr))
        print(f"Correct no edges: (V={V})", np.mean(corr_no_edges))
        print()


if __name__ == "__main__":
    main()

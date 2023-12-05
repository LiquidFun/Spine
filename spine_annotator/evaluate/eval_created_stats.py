from collections import defaultdict

import numpy as np

# 192 px
# STATS_PATH = "logs/2023-09-25_194839/stats.npz"

# 416 px
STATS_PATH = "logs/2023-09-26_094648/stats.npz"

# 320 px
# STATS_PATH = "logs/2023-09-26_105632/stats.npz" # No fix top
# STATS_PATH = "logs/2023-09-26_120239/stats.npz"


def main():
    stats = np.load(STATS_PATH, allow_pickle=True)["stats"]
    accumulated = defaultdict(int)
    for i, stat in enumerate(stats):
        if stat == "Error":
            print("Error")
            # for key in ["f1", "correct", "correct_no_edges"]:
            #     accumulated[key] += 0
        else:
            print(stat["correct_no_edges"])
            for key in stat:
                if key in ["f1", "correct", "correct_no_edges"]:
                    accumulated[key] += stat[key]
        # print(i, stat.keys())
    # correct = sum(stat["correct"] for stat in stats)
    # correct_no_edges = sum(stat["correct_no_edges"] for stat in stats)
    for key in accumulated:
        accumulated[key] /= len(stats)
        print(key, accumulated[key])


if __name__ == "__main__":
    main()

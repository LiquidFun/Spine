from collections import Counter, defaultdict

from matplotlib import pyplot as plt

from spine_segmentation.resources.preloaded import get_measure_statistics


def plot_nan_counts_by_column(stats):
    nan_counts = stats.dicom_metadata.isna().sum()
    # nan_counts = nan_counts[nan_counts > 0]
    nan_counts = nan_counts.sort_values(ascending=False)
    # print(nan_counts.to_string())
    nan_counts.plot.bar()
    plt.show()


def count_column_prefixes(stats):
    counts = Counter(column.split(";")[0] for column in stats.nan_filtered_statistics.columns)
    print(*counts.items(), sep="\n")


def print_metadata(stats):
    metadata = stats.dicom_metadata
    metadata = stats.statistics
    subset = metadata[:10]
    subset.loc[-1] = metadata.isna().sum()
    subset.loc[-2] = metadata.nunique()
    subset = subset.sort_index()
    # table = pd.concat([subset, nan_counts])
    print(subset.T.to_string())
    print(subset.T.to_csv("metadata.csv"))

    print(sorted(stats.statistics.columns.str.extract(r";(.*)")[0].unique()))

    print(subset.T[subset.T[-2] != 1].to_string())


def print_stat_columns(stats):
    keys = sorted(stats.statistics.columns.str.extract(r";(.*)")[0].unique())
    multikeys = defaultdict(set)
    prefixes = defaultdict(set)
    for key in keys:
        prefix, *middle, postfix = key.split("_")
        multikeys["_".join(middle)].add(postfix)
        prefixes["_".join(middle)].add(prefix)

    for key, value in multikeys.items():
        print(f"{{{','.join(prefixes[key])}}}_{key}_{{{','.join(value)}}}")

    # print(*, sep="\n")


def main():
    stats = get_measure_statistics()
    plot_nan_counts_by_column(stats)
    print_metadata(stats)
    print_stat_columns(stats)
    # count_column_prefixes(stats)


if __name__ == "__main__":
    main()

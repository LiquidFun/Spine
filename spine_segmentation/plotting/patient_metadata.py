from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from spine_segmentation.resources.preloaded import get_measure_statistics

plots_path = Path(__file__).absolute().parent / "plots" / "metadata"


def main():
    plots_path.mkdir(parents=True, exist_ok=True)

    stats = get_measure_statistics()
    metadata = stats.dicom_metadata
    columns = ["PatientAge", "PatientSex", "PatientWeight", "PatientSize", "InstitutionAddress", "StudyDate"]
    for column_postfix in columns:
        column_name = f"DICOM;{column_postfix}"
        print(column_name)
        data = metadata[column_name]

        if column_postfix == "PatientAge":
            data = data.str.replace("Y", "").astype(int)

        if column_postfix == "StudyDate":
            month = ((data // 100 % 100) - 1) / 12
            year = data // 10000
            data = year + month
            print(year, month)

        try:
            average = np.average(data)
        except:
            average = None

        # try:
        #     data.plot.hist(bins=np.unique(data))
        # except TypeError:
        #     counts = pd.DataFrame.from_dict(Counter(data), orient="index")
        #     counts.plot(kind="bar")
        # title = f"Histogram of {column_name}"
        # if average is not None:
        #     title += f" ({average:.2f})"

        if data.dtype.kind in "biufc":  # Types representing continuous data
            # For continuous data, use seaborn's histplot
            plt.figure(figsize=(10, 6))
            sns.histplot(data, bins=len(np.unique(data)), kde=False)
        else:
            # For categorical data, use seaborn's countplot or barplot
            plt.figure(figsize=(10, 6))
            if isinstance(data, pd.Series):
                sns.countplot(x=data)
            else:  # handling raw lists or arrays
                data_count = pd.DataFrame.from_dict(Counter(data), orient="index").reset_index()
                data_count.columns = ["category", "count"]
                sns.barplot(x="category", y="count", data=data_count)

        # Set title and labels
        title = f"Histogram of {column_name}"
        if average is not None:
            title += f" ({average:.2f})"
        plt.title(title)
        plt.xlabel(column_name.split(";")[-1])
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(plots_path / f"{column_name.split(';')[-1]}Plot.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    main()

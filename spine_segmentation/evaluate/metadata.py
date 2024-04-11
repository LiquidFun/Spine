from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy import stats

from spine_segmentation.datasets.metadata_prediction_dataset import MetadataPredictionModule
from spine_segmentation.models.pl_modules.regression_module import RegressionModule
from spine_segmentation.models.regression.regression_model import RegressionModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "<chkpt_path>"
    model = RegressionModule.load_from_checkpoint(path, device, model=RegressionModel(4)).eval()
    # model = lightning.pytorch.LightningModule.load_from_checkpoint(path, device).eval()
    # model = torch.load(path, device).eval()

    data_module = MetadataPredictionModule()
    val_data = data_module.val_dataset

    results = []
    for i in range(len(val_data)):
        print(i)
        sample = val_data[i]
        prediction = model.predict(sample.image)
        results.append(prediction)
        results.append(sample.gt)

    df = pd.DataFrame(results)
    df.to_csv("predictions.csv", index=False)


def compare_csv():
    path = "<csv_path>"
    pd.options.display.float_format = "{:.4f}".format
    np.set_printoptions(formatter={"float_kind": pd.options.display.float_format})
    df = pd.read_csv(path)
    df *= 100
    df.iloc[:, 2] /= 100
    df.iloc[:, 6] /= 100
    predicted = df.iloc[:, 1:5].to_numpy()
    correct = df.iloc[:, 5:9].to_numpy()
    plot(predicted, correct)
    # plot(predicted, "Prediction")

    print("Age | Sex | Size | Weight")
    data_variance = calculate_variance(0, correct)
    data_stddev = np.sqrt(data_variance)
    # data_confidence = confidence_interval(0, correct)
    print("Data variance", data_variance)
    print("Data stddev", data_stddev)
    # print("Confidence interval", data_confidence)
    # counts = (data_confidence[0] <= correct) & (correct <= data_confidence[1])
    # print(counts.sum(axis=0))
    # print(predicted)
    # print(correct)
    # diff = (predicted - correct)
    # mse = ((diff**2).mean(axis=0))
    saved = defaultdict(lambda: defaultdict(list))
    sep = "=" * 10
    print(f"{sep} Prediction {sep}")
    evaluate(predicted, correct, saved["EfficientNet"])

    print(f"{sep} Mean predictor {sep}")
    mean = correct.mean(axis=0)
    evaluate(mean, correct, saved["Mean_Predictor"])

    print(f"{sep} Median predictor {sep}")
    median = np.median(correct, axis=0)
    evaluate(median, correct, saved["Median_Predictor"])

    print_for_sheets(saved)


def print_for_sheets(saved):
    print("Copy this for sheets:\n")
    for predictor_name, saved_data in saved.items():
        row = [predictor_name, ""]
        for i in range(4):
            for metric in ["MSE", "Stddev", "MAE"]:
                row.append(str(saved_data[metric][i]))
        print("\t".join(row))


def evaluate(prediction, data, save_in):
    print(prediction)
    print("== MSE")
    print(mse := calculate_mse(prediction, data))
    save_in["MSE"] = mse
    print("== Variance")
    variance = calculate_variance(prediction, data)
    save_in["Variance"] = variance
    stddev = np.sqrt(variance)
    save_in["Stddev"] = stddev
    print(variance)
    print("== Stddev")
    print(stddev)
    print("== MAE")
    print(mae := calculate_mae(prediction, data))
    save_in["MAE"] = mae
    # print("== Confidence Interval 95%")
    # print(confidence_interval(prediction, data))
    print("\n")


def calculate_mse(prediction, data):
    mse = (data - prediction) ** 2
    return mse.mean(axis=0)


def calculate_mae(prediction, data):
    mae = np.abs(data - prediction)
    return mae.mean(axis=0)


def calculate_msd(prediction, data):
    """Mean signed difference"""
    msd = data - prediction
    return msd.mean(axis=0)


def calculate_variance(prediction, data):
    bias = calculate_msd(prediction, data)
    mse = calculate_mse(prediction, data)
    variance = mse - bias**2
    return variance


def confidence_interval(prediction, data, confidence=0.99):
    """Calculate confidence interval for given data and prediction"""
    # Only makes sense for normally distributed data!
    mean = calculate_mae(prediction, data)
    interval = stats.norm.interval(confidence, loc=mean, scale=stats.sem(data))
    return interval


def plot(predicted, data):
    plt.suptitle(f"Data and predictions distribution (n={data.shape[0]})")
    for i, title in zip(range(data.shape[1]), ["Age", "Sex", "Size", "Weight"]):
        plt.subplot(2, 2, i + 1)
        plt.title(title)
        # color = "blue" if title_suffix.lower() == "data" else "red"
        plt.hist(data[:, i], bins=50, color="blue", label="Data")
        plt.hist(predicted[:, i], bins=50, color="red", alpha=0.7, label="Prediction")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    compare_csv()
    # main()

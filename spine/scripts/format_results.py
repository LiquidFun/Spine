import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml


def last_key(config, location):
    if location == "":
        return config.split(".")[-1]
    rest = location.split(".", 1)[1] if "." in location else ""
    return last_key(config[location.split(".")[0]], rest)


def maybe_int(string):
    try:
        return int(string)
    except ValueError:
        return string


def split_into_numbers_and_text_tuple(string):
    res = re.split(r"(\d+|\D+)", string)
    return list(map(maybe_int, res))


def textbf_if_max(value, max_value, overall_max_value):
    string = f"{value:.3f}"
    with_star = "$^{\star}$" if round(value, 3) == overall_max_value else ""
    return (r"\textbf{" + string + "}" if round(value, 3) == max_value else string) + with_star


def with_citation(string):
    citation = ""
    for key, val in term_to_citation.items():
        if key.lower() in string.lower():
            citation = r" \cite{" + val + "}"
    marker = ""
    if "timm-" in string:
        marker = r"$^{\dagger}$"

    return string.replace("_", r"\_") + marker + citation


term_to_citation = {
    "Unet": "Ronneberger_2015",
    "DeepLabV3Plus": "Chen2018",
    "DeepLabV3": "Chen2017",
    "MAnet": "Fan2020",
    "ResNet": "He2015",
    "FPN": "Lin2016",
    "PAN": "Li2018",
    "DenseNet": "Huang2016",
    "EfficientNet": "Tan2019",
    "ResNeSt": "Zhang2020",
    "ResNeXt": "Xie2016",
    "VGG": "Simonyan2014",
    "Linknet": "Chaurasia2017",
    "mit": "Xie2021",
    "Res2Net": "Gao2021",
    "RegNetX": "Radosavovic2020",
    "GerNet": "VanHilten2021",
    "MobileOne": "Vasu2022",
    # "CrossEntropyLoss": "Goodfellow2016",
}


def main():
    csv = pd.read_csv(Path(__file__).absolute().parent / "task1_results/runs_all.csv")
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: (0, 0))))
    overall_max_dice = 0
    overall_max_iou = 0
    for row in csv.itertuples():
        with open(row.ConfigPath) as config_file:
            config = yaml.load(config_file, yaml.Loader)
            model = last_key(config, "model.init_args.model.class_path")
            encoder = last_key(config, "model.init_args.model.init_args.encoder_name")
            loss = last_key(config, "model.init_args.loss.class_path")
            augmentation = False
            try:
                last_key(config, "model.init_args.augmentation")
                augmentation = True
            except (AttributeError, KeyError):
                pass
            # print(augmentation)
            dice = row.val_f1
            iou = row.val_iou
            if not math.isnan(dice) and not math.isnan(iou):
                prev_dice, prev_iou = table[model][encoder][loss]
                table[model][encoder][loss] = (max(prev_dice, dice), max(prev_iou, iou))
            overall_max_dice = max(overall_max_dice, round(dice, 3))
            overall_max_iou = max(overall_max_iou, round(iou, 3))
            # print(model, encoder, loss)

    # for model in table:
    #     print(model)
    #     for encoder in table[model]:
    #         print(f"\t{encoder}")
    #         for loss in table[model][encoder]:
    #             print(f"\t\t{loss}: {table[model][encoder][loss]}")
    #         print()

    print(r"\hline")
    print(" & ".join(["Section", "Architecture", "Encoder", "Loss", "\Ac{DSC}", "\Ac{IoU}"]) + r" \\\hline\hline")
    for model in table:
        max_dice = 0
        max_iou = 0
        for encoder in sorted(table[model], key=split_into_numbers_and_text_tuple):
            for loss in table[model][encoder]:
                max_dice = max(max_dice, round(table[model][encoder][loss][0], 3))
                max_iou = max(max_iou, round(table[model][encoder][loss][1], 3))

        for encoder in sorted(table[model], key=split_into_numbers_and_text_tuple):
            # if "JaccardLoss" in table[model][encoder] and "DiceLoss" in table[model][encoder]:
            for loss in table[model][encoder]:
                dice, iou = table[model][encoder][loss]
                line = [
                    r"\ref{sec:slice-based-segmentation}",
                    with_citation(model),
                    with_citation(encoder),
                    with_citation(loss),
                    textbf_if_max(dice, max_dice, overall_max_dice),
                    textbf_if_max(iou, max_iou, overall_max_iou),
                ]
                joined = " & ".join(line) + r" \\"
                print(joined)
        print(r"\hline")


if __name__ == "__main__":
    main()

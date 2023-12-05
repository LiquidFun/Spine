import itertools
import os
import random
from datetime import datetime
from pathlib import Path

import yaml

run_configs_path = Path(__file__).absolute().parent.parent / "run_configs"

# Define your parts to exchange here
PARTS_TO_EXCHANGE_MODEL_LOSS = {
    "model.init_args.model.class_path": [
        "segmentation_models_pytorch.Unet",
        # "segmentation_models_pytorch.UnetPlusPlus",
        "segmentation_models_pytorch.FPN",
        # "segmentation_models_pytorch.PAN",
        "segmentation_models_pytorch.MAnet",
        # "segmentation_models_pytorch.PSPNet",
        # "segmentation_models_pytorch.Linknet",
        # "segmentation_models_pytorch.DeepLabV3",
        # "segmentation_models_pytorch.DeepLabV3Plus",
    ],
    "model.init_args.loss": [
        {
            "class_path": "segmentation_models_pytorch.losses.DiceLoss",
            "init_args": {
                "mode": "multiclass",
                "log_loss": True,
                "classes": [1, 2],
            },
        },
        # {
        #     "class_path": "segmentation_models_pytorch.losses.FocalLoss",
        #     "init_args": {
        #         "mode": "multiclass",
        #         "ignore_index": 0,
        #     },
        # },
        # {
        #     "class_path": "segmentation_models_pytorch.losses.LovaszLoss",
        #     "init_args": {
        #         "mode": "multiclass",
        #         "ignore_index": 0,
        #     },
        # },
        # {
        #     "class_path": "segmentation_models_pytorch.losses.JaccardLoss",
        #     "init_args": {"mode": "multiclass", "classes": [1, 2]},
        # },
        # {
        #     "class_path": "torch.nn.CrossEntropyLoss",
        # },
    ],
    "model.init_args.augmentation": [
        {
            "class_path": "spine_annotator.datasets.augmentation.ImageAugmentation",
            "init_args": {
                "cropping_type": "none",
                "geometric_aug_type": "all",
                "photometric_aug_type": "all",
            },
        },
        None,
    ],
    # "data.init_args.each_roi_is_separate_class": [True, False],
}

PARTS_TO_EXCHANGE_3D_MODEL_LOSS = {
    "model.init_args.model": [
        {
            "class_path": "monai.networks.nets.UNet",
            "init_args": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 3,
                "channels": [128, 256, 512, 1024, 2048],
                "strides": [2, 2, 2, 2],
                "num_res_units": 2,
                "dropout": 0.5,
            },
        },
        {
            "class_path": "monai.networks.nets.UNet",
            "init_args": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 3,
                "channels": [128, 256, 512, 1024, 2048],
                "strides": [2, 2, 2, 2],
                "num_res_units": 2,
                "dropout": 0.0,
            },
        },
    ],
    "model.init_args.loss": [
        {
            "class_path": "segmentation_models_pytorch.losses.DiceLoss",
            "init_args": {
                "mode": "multiclass",
                "log_loss": True,
                "classes": [1, 2],
            },
        },
        {
            "class_path": "segmentation_models_pytorch.losses.FocalLoss",
            "init_args": {
                "mode": "multiclass",
                "ignore_index": 0,
            },
        },
        {
            "class_path": "segmentation_models_pytorch.losses.LovaszLoss",
            "init_args": {
                "mode": "multiclass",
                "ignore_index": 0,
            },
        },
        {
            "class_path": "segmentation_models_pytorch.losses.JaccardLoss",
            "init_args": {"mode": "multiclass", "classes": [1, 2]},
        },
        {
            "class_path": "torch.nn.CrossEntropyLoss",
        },
    ],
    # "model.init_args.augmentation": [
    #     {
    #         "class_path": "spine_annotator.datasets.augmentation.ImageAugmentation",
    #         "init_args": {
    #             "cropping_type": "none",
    #             "geometric_aug_type": "all",
    #             "photometric_aug_type": "all",
    #         },
    #     },
    #     None,
    # ],
    # "data.init_args.each_roi_is_separate_class": [True, False],
}


PARTS_TO_EXCHANGE_ENCODER = {
    "model.init_args.model.init_args.encoder_name": [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x4d",
        "resnext101_32x8d",
        "timm-resnest14d",
        "timm-resnest26d",
        "timm-resnest50d",
        "timm-resnest101e",
        "timm-resnest200e",
        "timm-res2net50_26w_4s",
        "timm-res2net101_26w_4s",
        "timm-res2net50_26w_6s",
        "timm-res2net50_26w_8s",
        "timm-res2net50_48w_2s",
        "timm-res2net50_14w_8s",
        "timm-res2next50",
        "timm-regnetx_002",
        "timm-regnetx_004",
        "timm-regnetx_006",
        "timm-regnetx_008",
        "timm-regnetx_016",
        "timm-regnetx_032",
        "timm-regnetx_040",
        "timm-regnetx_064",
        "timm-regnetx_080",
        "timm-regnetx_120",
        "timm-regnetx_160",
        "timm-gernet_s",
        "timm-gernet_m",
        "timm-gernet_l",
        "se_resnet50",
        "se_resnet101",
        "se_resnet152",
        "se_resnext50_32x4d",
        "se_resnext101_32x4d",
        "timm-skresnet18",
        "timm-skresnet34",
        "timm-skresnext50_32x4d",  # pragma: allowlist secret
        "densenet121",
        "densenet169",
        "densenet201",
        "densenet161",
        "inceptionresnetv2",
        "inceptionv4",
        "xception",
        "efficientnet-b0",
        "efficientnet-b1",
        "efficientnet-b2",
        "efficientnet-b3",
        "efficientnet-b4",
        "efficientnet-b5",
        "efficientnet-b6",
        "efficientnet-b7",
        "timm-efficientnet-b0",
        "timm-efficientnet-b1",
        "timm-efficientnet-b2",
        "timm-efficientnet-b3",
        "timm-efficientnet-b4",
        "timm-efficientnet-b5",
        "timm-efficientnet-b6",
        "timm-efficientnet-b7",
        "timm-efficientnet-b8",
        "dpn68",
        "dpn68b",
        "dpn92",
        "dpn98",
        "dpn107",
        "dpn131",
        "vgg11",
        "vgg11_bn",
        "vgg13",
        "vgg13_bn",
        "vgg16",
        "vgg16_bn",
        "vgg19",
        "vgg19_bn",
        "mit_b0",
        "mit_b1",
        "mit_b2",
        "mit_b3",
        "mit_b4",
        "mit_b5",
        "mobileone_s0",
        "mobileone_s1",
        "mobileone_s2",
        "mobileone_s3",
        "mobileone_s4",
    ],
}

PARTS_TO_EXCHANGE = PARTS_TO_EXCHANGE_3D_MODEL_LOSS

# import segmentation_models_pytorch as smp
# d = smp.losses.LovaszLoss()


def read_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def write_yaml(data, file_path):
    with open(file_path, "w") as file:
        yaml.dump(data, file)


def exchange_parts(config, exchange_with=None):
    for key, options in PARTS_TO_EXCHANGE.items():
        parts = key.split(".")
        value = config
        for part in parts[:-1]:
            value = value.get(part, None)
            if value is None:
                break
        if value is not None:
            if exchange_with is None:
                value[parts[-1]] = random.choice(options)
            else:
                value[parts[-1]] = exchange_with[key]
    return config


def generate_configs(n, template_file):
    template = read_yaml(template_file)
    # encoder = template["model"]["init_args"]["model"]["init_args"]["encoder_name"]
    model_name = template["model"]["init_args"]["model"]["class_path"].split(".")[-1]
    model_name = "3D_Unet"

    experiment_name = "seg_volume"

    template["model"]["init_args"]["experiment_name"] = f"spine_annotator_{experiment_name}"
    template["model"]["init_args"]["run_prefix"] = f"Seg{model_name}"
    # template["model"]["init_args"]["experiment_name"] = f"spine_annotator_seg_{model_name}"

    today = datetime.now().strftime("%Y-%m-%d")
    # dir_name = run_configs_path / f"experiments_{today}_{model_name}"
    dir_name = run_configs_path / f"experiments_{today}_{experiment_name}_2"
    dir_name.mkdir(exist_ok=True)

    # for i in range(n):
    #     config = exchange_parts(template.copy())
    #     config_file = os.path.join(dir_name, f"config_{i + 1}.yaml")
    #     write_yaml(config, config_file)

    combinations = list(itertools.product(*PARTS_TO_EXCHANGE.values()))

    for i, comb in enumerate(combinations):
        as_dict = dict(zip(PARTS_TO_EXCHANGE.keys(), comb))
        # print(as_dict)
        config = exchange_parts(template.copy(), as_dict)
        config_file = os.path.join(dir_name, f"config_{i + 1:03}.yaml")
        print(config_file)
        write_yaml(config, config_file)


def main():
    # generate_configs(10, run_configs_path / "segmentation.yaml")
    generate_configs(10, run_configs_path / "volume_segmentation.yaml")


# Usage
if __name__ == "__main__":
    main()

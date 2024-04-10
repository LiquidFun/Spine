from typing import Literal

import torch
from kornia import augmentation as K
from torch import Tensor


class ImageAugmentation(torch.nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(
        self,
        cropping_type: Literal["none", "same", "many"] = "many",
        geometric_aug_type: Literal["none", "all"] = "all",
        photometric_aug_type: Literal["none", "all"] = "all",
        use_mask=True,
    ) -> None:
        super().__init__()

        self.use_mask = use_mask

        # === GEOMETRIC ===
        geometric_augmentations = []

        if geometric_aug_type == "all":
            geometric_augmentations += [
                K.RandomVerticalFlip(),  # Note that patient is lying on back, so this is actually a horizontal flip
                K.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), p=0.50),
            ]

        if cropping_type == "same":
            geometric_augmentations += [
                K.RandomCrop((320, 300), p=1.0),
                K.PadTo((320, 896)),
            ]
        if cropping_type == "many":
            geometric_augmentations += [
                K.RandomCrop((320, 300), p=0.1),
                K.RandomCrop((320, 400), p=0.15),
                K.RandomCrop((320, 500), p=0.2),
                K.RandomCrop((320, 600), p=0.2),
                K.RandomCrop((320, 700), p=0.2),
                K.PadTo((320, 896)),
            ]

        self.geometric_augmentations = K.AugmentationSequential(
            *geometric_augmentations, data_keys=(["input", "mask"] if use_mask else ["input"])
        )

        # === PHOTOMETRIC ===
        photometric_augmentations = []
        if photometric_aug_type == "all":
            photometric_augmentations += [
                K.RandomBrightness((0.9, 1.1), p=0.25),
                K.RandomContrast((0.9, 1.1), p=0.25),
                K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.25),
                K.RandomGaussianNoise(0.0, 0.01, p=0.25),
                K.RandomGaussianNoise(0.0, 0.02, p=0.15),
                K.RandomGaussianNoise(0.0, 0.03, p=0.05),
            ]

        self.photometric_augmentations = K.AugmentationSequential(
            # K.RandomGamma((0.9, 1.1), (1.5, 1.5), p=0.5),
            data_keys=["input"],
        )

    @torch.no_grad()
    def forward(self, inputs: Tensor, mask: Tensor = None):
        if self.use_mask:
            inputs, mask = self.geometric_augmentations(inputs, mask)
        else:
            inputs = self.geometric_augmentations(inputs)
        inputs = self.photometric_augmentations(inputs)
        if self.use_mask:
            return inputs, mask
        return inputs

    def single_image(self, inputs: Tensor, mask: Tensor):
        inputs, mask = self.forward(inputs[None], mask[None])
        return inputs[0], mask[0]

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel import Nifti1Image

from spine_segmentation.instance_separation.instance_separation import separate_segmented_rois

if __name__ == "__main__":
    data = Path("<path>")
    save_to = Path("<path>")
    save_to.mkdir(exist_ok=True)


class Spine:
    def __init__(self, path):
        pid = path.name
        # Load 3 nifti files using nibabel
        self.image = nib.load(path / f"{pid}.nii.gz")
        self.wk = nib.load(path / f"{pid}_WK.nii.gz")
        self.bs = nib.load(path / f"{pid}_NR.nii.gz")
        assert self.image.shape == self.wk.shape == self.bs.shape, "Shapes do not match"
        # print(self.image.shape, self.wk.shape, self.bs.shape, np.unique(self.wk.dataobj), np.unique(self.bs.dataobj))
        self.seg = separate_segmented_rois(self.wk.get_fdata())

        full = self.get_full_with_spn_classes()
        cropped_size = self.get_cropped_size(full)
        self.cropped_full = full[:, :, :cropped_size]
        self.cropped_wk = self.wk.get_fdata()[:, :, :cropped_size]
        self.cropped_bs = self.bs.get_fdata()[:, :, :cropped_size]
        self.cropped_image = self.image.get_fdata()[:, :, :cropped_size]

        for i, obj in enumerate([self.cropped_image, self.cropped_image + self.cropped_full * 100, self.cropped_full]):
            plt.subplot(3, 1, i + 1)
            plt.imshow(obj[9, :, ::-1].T)
        plt.show()

    def save_to(self, index):
        mask_path = save_to / "Mask"
        mask_path.mkdir(exist_ok=True)
        mr_path = save_to / "MR"
        mr_path.mkdir(exist_ok=True)
        Nifti1Image(self.cropped_image, self.image.affine, self.image.header).to_filename(
            mr_path / f"Case{index}.nii.gz"
        )
        Nifti1Image(self.cropped_full, self.bs.affine, self.bs.header).to_filename(
            mask_path / f"mask_case{index}.nii.gz"
        )

    def get_cropped_size(self, full: np.ndarray):
        up_to = np.nonzero(full)[2].max() - 5
        # cropped = full[:, :, :up_to]
        return up_to

    def get_full_with_spn_classes(self) -> np.ndarray:
        bs, wk = self.bs.get_fdata(), self.seg
        wk_count = 10
        bs_count = 9
        wk[wk <= wk.max() - wk_count] = 0
        bs[bs <= bs.max() - bs_count] = 0
        # print(np.unique(bs), np.unique(wk))

        wk[wk != 0] -= wk[wk != 0].max() + 1
        bs[bs != 0] -= bs[bs != 0].max() + 1
        wk *= -1
        bs *= -1

        # print(np.unique(bs), np.unique(wk))

        bs[bs != 0] += wk_count
        wk[bs != 0] = 0
        full = (wk + bs).astype(int)
        return full
        # print(np.unique(full))
        # plt.imshow(full[9, :, ::-1].T)
        # plt.show()


def main():
    blacklist_num = ["107132"]
    # spines = []
    index = 1
    dirs = list(data.glob("*"))
    print(len(dirs))
    for i, patient_dir in enumerate(dirs):
        print(f"Processing {i=} {index=} {patient_dir.name}")
        if patient_dir.name in blacklist_num:
            continue
        try:
            if not (save_to / "Mask" / f"mask_case{index}.nii.gz").exists():
                spine = Spine(patient_dir)
                spine.save_to(index)
            # spines.append(spine)
            index += 1
        except FileNotFoundError as e:
            print(e)
            continue
    # print(len(spines))


if __name__ == "__main__":
    main()

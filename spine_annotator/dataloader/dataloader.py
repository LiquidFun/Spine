from pathlib import Path
from typing import List

import nibabel as nib
from loguru import logger

from spine_annotator.spine_types.spine import Spine

DATA_PATH = Path.home() / "DataSpine"


class DataLoader:
    def load_default(self):
        self.load_ct()
        self.load_mri()
        return self

    def load_ct(self):
        self.spines.extend(self._get_verse_dataset(DATA_PATH / "verse19"))
        self.spines.extend(self._get_verse_dataset(DATA_PATH / "verse20"))
        return self

    def load_mri(self):
        pass
        # self.spines.extend(self._get_verse_dataset(DATA_PATH / "verse19"))
        # self.spines.extend(self._get_verse_dataset(DATA_PATH / "verse20"))
        return self

    def __init__(self):
        self.spines = []

    def _get_verse_dataset(self, path: Path) -> List[Spine]:
        data_path = path / "rawdata"
        derivatives_path = path / "derivatives"
        spines = []
        if data_path.exists() and derivatives_path.exists():
            for nifti_path in data_path.rglob("*.nii.gz"):
                segmentation_path = (
                    derivatives_path / nifti_path.parts[-2] / nifti_path.name.replace("_ct", "_seg-vert_msk")
                )
                assert segmentation_path.exists()
                data = nib.load(nifti_path)
                segmentation = nib.load(segmentation_path)
                assert data.shape == segmentation.shape
                spines.append(Spine(data, segmentation))
        logger.info(f"Successfully loaded {len(spines)} from {path}")
        return spines

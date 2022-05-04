import typing as t
from pathlib import Path

import cv2
import pandas as pd
import torch
import torchvision


class HelenFaceMorphingDataset(torchvision.datasets.VisionDataset):
    def __init__(self, dataset_root: str):
        super().__init__(root=dataset_root)

        self.images_path = self._load_images()
        self.keypoints_df = pd.read_csv(Path(self.root) / "metadata.csv")

    def __len__(self) -> int:
        return len(self.images_path)

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images_path[index]
        image, keypoints = self._load_sample(image_path)

        return image, keypoints

    def _load_images(self) -> t.List[str]:
        images_dir = Path(self.root) / "images"

        images = sorted(list(images_dir.glob("*.jpg")))

        return images

    def _load_sample(self, image_path: Path) -> t.Tuple[str, torch.Tensor]:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        keypoints = self.keypoints_df.loc[
            self.keypoints_df["Filenames"] == image_path.name.rsplit(".", 1)[0]
        ]
        keypoints = keypoints.iloc[0]

        return image, keypoints

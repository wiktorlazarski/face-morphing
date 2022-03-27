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
        pass

    def __len__(self) -> int:
        return len(self.images_path)

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images_path[index]
        image, key_points = self._load_sample(image_path)

        return image, key_points

    def _load_images(self) -> t.List[str]:
        images_dir = Path(self.root) / "images"

        images = sorted(list(images_dir.glob("*.jpg")))

        return images

    def _load_sample(self, image_path: Path) -> t.Tuple[str, torch.Tensor]:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        key_points_csv = pd.read_csv(image_path.parent.parent / "metadata.csv")

        key_points = key_points_csv.loc[
            key_points_csv["Filenames"] == image_path.name.rsplit(".", 1)[0]
        ]
        key_points = key_points.iloc[0]

        return image, key_points

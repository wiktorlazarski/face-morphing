import typing as t
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision

import face_morphing.image_processing as ip
import scripts.training.augmentations as aug


class HelenFaceMorphingDataset(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        dataset_root: str,
        preprocess_pipeline: t.Optional[ip.PreprocessingPipeline] = None,
        augmentation_pipeline: t.Optional[aug.AugmentationPipeline] = None,
    ):
        super().__init__(root=dataset_root)

        self.images_path = self._load_images()
        self.keypoints_df = pd.read_csv(Path(self.root) / "metadata.csv")
        self.preprocess_pipeline = preprocess_pipeline
        self.augmentation_pipeline = augmentation_pipeline

    def __len__(self) -> int:
        return len(self.images_path)

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images_path[index]
        image, keypoints = self._load_sample(image_path)

        if self.augmentation_pipeline is not None:
            image, keypoints = self.augmentation_pipeline(
                image=image, keypoints=keypoints
            )

        if self.preprocess_pipeline is not None:
            image, keypoints = self.preprocess_pipeline.preprocess_image_keypoints(
                image=image, keypoints=keypoints
            )

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
        keypoints = self._format_keypoints(image, keypoints)

        return image, keypoints

    def _format_keypoints(
        self, image: np.ndarray, keypoints: torch.Tensor
    ) -> t.List[t.Tuple[int, int]]:
        keypoints_list = []
        size_x = image.shape[0]
        size_y = image.shape[1]
        for iter in range(1, int((len(keypoints) - 1) / 2)):
            if keypoints[2 * iter - 1] < 0.0:
                keypoints[2 * iter - 1] = 0.0
            elif keypoints[2 * iter - 1] > size_x:
                keypoints[2 * iter - 1] = size_x
            if keypoints[2 * iter] < 0.0:
                keypoints[2 * iter] = 0.0
            elif keypoints[2 * iter] > size_y:
                keypoints[2 * iter] = size_y

            keypoints_list.append((keypoints[2 * iter - 1], keypoints[2 * iter]))

        return keypoints_list

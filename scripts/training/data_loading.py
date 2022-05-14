import typing as t
from pathlib import Path

import cv2
import os
import numpy as np
import pandas as pd
import torch
import torchvision
from facenet_pytorch import MTCNN

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
        self.mtcnn = MTCNN(min_face_size=100, thresholds=[0.7707,  0.8219,  0.8137])

    def __len__(self) -> int:
        return len(self.images_path)

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images_path[index]
        image, keypoints = self._load_sample(image_path)
        image, keypoints = self._crop_face(image, keypoints)

        if self.augmentation_pipeline is not None:
            image, keypoints = self.augmentation_pipeline(
                image=image, keypoints=keypoints
            )

        if self.preprocess_pipeline is not None:
            image, keypoints = self.preprocess_pipeline.preprocess_sample(
                image=image, keypoints=keypoints
            )

        return image, keypoints

    def _crop_face(self, img, keypoints):
        box, _ = self.mtcnn.detect(img)
        sy, sx, _ = img.shape
        best = 0
        best_face = None
        for face in box:
            x1 = np.clip(np.round(face[0]).astype(dtype=np.int32), a_min=0, a_max=sx)
            y1 = np.clip(np.round(face[1]).astype(dtype=np.int32), a_min=0, a_max=sy)
            x2 = np.clip(np.round(face[2]).astype(dtype=np.int32), a_min=0, a_max=sx)
            y2 = np.clip(np.round(face[3]).astype(dtype=np.int32), a_min=0, a_max=sy)

            current = sum([ 1 for x, y in keypoints if x1 <= x and x <= x2 and y1 <= y and y <= y2 ])
            if current > best:
                best = current
                best_face = (x1, x2, y1, y2)

        x1, x2, y1, y2 = best_face
        w = x2 - x1
        h = y2 - y1
        padding_w, padding_h = int(0.1 * w), int(0.1 * h)
        x1 = max(x1 - padding_w, 0)
        y1 = max(y1 - padding_h, 0)
        x2 = min(x2 + padding_w, sx)
        y2 = min(y2 + padding_h, sy)

        cropped = img[y1:y2, x1:x2, :]
        points = [
            [
                a - x1,
                b - y1
            ] for a, b in keypoints
        ]
        return cropped, points

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
        size_y = image.shape[0]
        size_x = image.shape[1]
        for iter in range(1, (len(keypoints) - 1) // 2, 4):
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

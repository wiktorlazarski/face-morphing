import typing as t

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import cv2
import os

class PreprocessingPipeline:

    IMAGENET_MEANS = [0.485, 0.456, 0.406]
    IMAGENET_STDS = [0.229, 0.224, 0.225]

    def __init__(self, nn_image_input_resolution: int) -> None:
        self.nn_image_input_resolution = nn_image_input_resolution
        self.image_preprocessing_pipeline = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (nn_image_input_resolution, nn_image_input_resolution)
                ),
                transforms.Normalize(
                    mean=PreprocessingPipeline.IMAGENET_MEANS,
                    std=PreprocessingPipeline.IMAGENET_STDS,
                ),
            ]
        )


    def __call__(
        self, image: np.ndarray, keypoints: torch.Tensor
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        return self.preprocess_image_keypoints(image, keypoints)

    def preprocess_sample(
        self, image: np.ndarray, keypoints: torch.Tensor
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        resized_image = self.image_preprocessing_pipeline(Image.fromarray(image))

        for iter in range(int(len(keypoints))):
            keypoints[iter] = list(keypoints[iter])
            keypoints[iter][0] = keypoints[iter][0] / image.shape[1]
            keypoints[iter][1] = keypoints[iter][1] / image.shape[0]

        return resized_image, torch.tensor(keypoints).float()

    def preprocess_image(
        self, image: np.ndarray
    ) -> torch.Tensor:
        resized_image = self.image_preprocessing_pipeline(Image.fromarray(image))

        return resized_image


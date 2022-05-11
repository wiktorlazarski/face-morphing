import typing as t

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


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
        resized_size_x = image.shape[0] / self.nn_image_input_resolution
        resized_size_y = image.shape[1] / self.nn_image_input_resolution

        for iter in range(1, int(len(keypoints))):
            keypoints[iter] = list(keypoints[iter])
            keypoints[iter][0] = keypoints[iter][0] * resized_size_x
            keypoints[iter][1] = keypoints[iter][1] * resized_size_y

        return resized_image, torch.tensor(keypoints).float()

import abc
import typing as t

import cv2
import numpy as np


class Morphing(abc.ABC):
    def __init__(self, num_morphs: int):
        self.num_morphs = num_morphs

    @abc.abstractmethod
    def generate_morph_sequence(
        self, from_img: np.ndarray, to_img: np.ndarray
    ) -> t.List[np.ndarray]:
        pass

    def looped_morphing(
        self, from_img: np.ndarray, to_img: np.ndarray
    ) -> t.List[np.ndarray]:
        from_to_morphs = self.generate_morph_sequence(from_img, to_img)
        to_from_morphs = self.generate_morph_sequence(to_img, from_img)

        return [*from_to_morphs, *to_from_morphs]

    def _resize_second_image(
        self, second_img: np.ndarray, first_image_res: t.Tuple[int, int]
    ) -> np.ndarray:
        resized_img = cv2.resize(second_img, first_image_res)
        return resized_img

    def _compute_morph(
        self, from_img: np.ndarray, to_img: np.ndarray, alpha: float
    ) -> np.ndarray:
        assert 0.0 <= alpha <= 1.0
        return cv2.addWeighted(
            src1=from_img, alpha=(1.0 - alpha), src2=to_img, beta=alpha, gamma=0.0
        )

import abc
import typing as t

import cv2
import numpy as np


class Morphing(abc.ABC):
    def __init__(self, num_morphs: int):
        self.num_morphs = num_morphs

    @abc.abstractmethod
    def generate_morph_sequence(
        self, from_img: np.ndarray, to_img: np.ndarray, *args: t.Any
    ) -> t.List[np.ndarray]:
        pass

    @abc.abstractmethod
    def looped_morphing(
        self, from_img: np.ndarray, to_img: np.ndarray
    ) -> t.List[np.ndarray]:
        pass

    def _resize_second_image(
        self,
        second_img: np.ndarray,
        first_image_res: t.Tuple[int, int],
        second_kps: t.Optional[t.List[t.Tuple[float]]] = None,
    ) -> np.ndarray:
        resized_img = cv2.resize(second_img, first_image_res)

        resized_kps = []
        if second_kps is not None:
            s_h, s_w = second_img.shape[:2]
            f_w, f_h = first_image_res

            sx, sy = f_w / s_w, f_h / s_h

            for kp in second_kps:
                x, y = kp
                new_x = x * sx
                new_y = y * sy
                resized_kps.append((round(new_x, 2), round(new_y, 2)))

        return resized_img, resized_kps

    def _compute_morph(
        self, from_img: np.ndarray, to_img: np.ndarray, alpha: float
    ) -> np.ndarray:
        assert 0.0 <= alpha <= 1.0
        return cv2.addWeighted(
            src1=from_img, alpha=(1.0 - alpha), src2=to_img, beta=alpha, gamma=0.0
        )

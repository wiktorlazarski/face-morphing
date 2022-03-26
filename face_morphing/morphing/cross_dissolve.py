import typing as t

import numpy as np

import face_morphing.morphing.algorithm as alg


class CrossDissolveMorphing(alg.Morphing):
    def __init__(self, num_morphs: int):
        super().__init__(num_morphs)

    def generate_morph_sequence(
        self, from_img: np.ndarray, to_img: np.ndarray
    ) -> t.List[np.ndarray]:
        from_height, from_width = from_img.shape[:2]
        resized_to_img, _ = self._resize_second_image(to_img, (from_width, from_height))

        morphs = []
        for alpha in np.linspace(0.0, 1.0, self.num_morphs):
            morph = self._compute_morph(from_img, resized_to_img, alpha=alpha)
            morphs.append(morph)

        return morphs

    def looped_morphing(
        self, from_img: np.ndarray, to_img: np.ndarray
    ) -> t.List[np.ndarray]:
        morphs = self.generate_morph_sequence(from_img, to_img)

        return [*morphs, *reversed(morphs)]

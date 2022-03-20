import typing as t

import cv2
import numpy as np

import face_morphing.morphing.algorithm as alg


class KeypointsAlignmentMorphing(alg.Morphing):
    def __init__(self, num_morphs: int):
        super().__init__(num_morphs)

    def generate_morph_sequence(
        self,
        from_img: np.ndarray,
        to_img: np.ndarray,
        from_kps: t.List[t.Tuple[float, float]],
        to_kps: t.List[t.Tuple[float, float]],
        combined_warped: bool = True,
    ) -> t.List[np.ndarray]:
        from_height, from_width = from_img.shape[:2]
        resized_to_img, resized_kps = self._resize_second_image(
            second_img=to_img,
            first_image_res=(from_width, from_height),
            second_kps=to_kps,
        )

        return (
            self._generate_morphs_combined_warped(
                from_img, resized_to_img, from_kps, resized_kps
            )
            if combined_warped
            else self._generate_morphs_not_combined_warped(
                from_img, resized_to_img, from_kps, resized_kps
            )
        )

    def looped_morphing(
        self,
        from_img: np.ndarray,
        to_img: np.ndarray,
        from_kps: t.List[t.Tuple[float, float]],
        to_kps: t.List[t.Tuple[float, float]],
        combined_warped: bool = True,
    ) -> t.List[np.ndarray]:
        morphs = self.generate_morph_sequence(
            from_img, to_img, from_kps, to_kps, combined_warped=combined_warped
        )
        return [*morphs, *reversed(morphs)]

    def _generate_morphs_combined_warped(
        self,
        from_img: np.ndarray,
        resized_to_img: np.ndarray,
        from_kps: t.List[t.Tuple[float, float]],
        resized_kps: t.List[t.Tuple[float, float]],
    ) -> t.List[np.ndarray]:
        from_height, from_width = from_img.shape[:2]

        holography_mat_from_to, _ = cv2.findHomography(
            np.array(resized_kps), np.array(from_kps)
        )
        warped_to_image = cv2.warpPerspective(
            resized_to_img, holography_mat_from_to, (from_width, from_height)
        )

        holography_mat_to_from, _ = cv2.findHomography(
            np.array(from_kps), np.array(resized_kps)
        )
        warped_from_image = cv2.warpPerspective(
            from_img, holography_mat_to_from, (from_width, from_height)
        )

        morphs = []
        for alpha in np.linspace(0.0, 1.0, self.num_morphs):
            from_prime = self._compute_morph(from_img, warped_from_image, alpha)
            to_prime = self._compute_morph(warped_to_image, resized_to_img, alpha)

            morph = self._compute_morph(from_prime, to_prime, alpha=alpha)
            morphs.append(morph)

        return morphs

    def _generate_morphs_not_combined_warped(
        self,
        from_img: np.ndarray,
        resized_to_img: np.ndarray,
        from_kps: t.List[t.Tuple[float, float]],
        resized_kps: t.List[t.Tuple[float, float]],
    ) -> t.List[np.ndarray]:
        from_height, from_width = from_img.shape[:2]

        holography_mat, _ = cv2.findHomography(
            np.array(resized_kps), np.array(from_kps)
        )
        warped_to_image = cv2.warpPerspective(
            resized_to_img, holography_mat, (from_width, from_height)
        )

        morphs = []
        for alpha in np.linspace(0.0, 1.0, self.num_morphs):
            morph = self._compute_morph(from_img, warped_to_image, alpha=alpha)
            morphs.append(morph)

        return morphs


################# test program ######################################

if __name__ == "__main__":
    import os

    a = KeypointsAlignmentMorphing(50)

    F_SAMPLE_ANN = os.path.join("data", "morph_samples", "1.txt")
    F_SAMPLE_IMG = os.path.join("data", "morph_samples", "1.jpg")

    S_SAMPLE_ANN = os.path.join("data", "morph_samples", "2.txt")
    S_SAMPLE_IMG = os.path.join("data", "morph_samples", "2.jpg")

    def load_sample(ann_file, img_file):
        keypoints = []

        with open(ann_file, "r") as ann_file:
            for i, line in enumerate(ann_file.readlines()):
                if i == 0:
                    continue

                coords = [float(val) for val in line.split(",")]
                keypoints.append(tuple(coords))

        img = cv2.imread(img_file)

        return img, keypoints

    from_img, from_kps = load_sample(F_SAMPLE_ANN, F_SAMPLE_IMG)
    to_img, to_kps = load_sample(S_SAMPLE_ANN, S_SAMPLE_IMG)

    seq = a.looped_morphing(from_img, to_img, from_kps, to_kps, combined_warped=False)

    while True:
        for morph in seq:
            cv2.imshow("homography morphing", morph)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                exit(0)

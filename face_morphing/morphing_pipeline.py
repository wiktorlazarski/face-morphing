import typing as t

import numpy as np
import torch
from facenet_pytorch import MTCNN

import face_morphing.constants as C
import face_morphing.image_processing as ip
import face_morphing.model as mdl


class FaceMorphingPipeline:
    def __init__(
        self, model_path: str = C.MODEL_PATH, nn_image_input_resolution: int = 256
    ):
        self._preprocessing_pipeline = ip.PreprocessingPipeline(
            nn_image_input_resolution=nn_image_input_resolution
        )
        self._model = mdl.KeyPointsModel.load_from_checkpoint(ckpt_path=model_path)
        self._model.eval()
        self.mtcnn = MTCNN(min_face_size=100, thresholds=[0.7707, 0.8219, 0.8137])

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.predict(image)

    def predict(self, image: np.ndarray) -> np.ndarray:
        preprocessed_image, (x, y), (w, h) = self._preprocess_image(image)
        mdl_out = self._model(preprocessed_image).detach()
        pred_segmap = self._postprocess_model_output(
            mdl_out,
            preprocessed_image_shape=(w, h),
            top_left_face_pointsce_points=(x, y),
        )
        return pred_segmap

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        box, _ = self.mtcnn.detect(image)
        sy, sx, _ = image.shape
        face = box[-1]

        x1 = np.clip(np.round(face[0]).astype(dtype=np.int32), a_min=0, a_max=sx)
        y1 = np.clip(np.round(face[1]).astype(dtype=np.int32), a_min=0, a_max=sy)
        x2 = np.clip(np.round(face[2]).astype(dtype=np.int32), a_min=0, a_max=sx)
        y2 = np.clip(np.round(face[3]).astype(dtype=np.int32), a_min=0, a_max=sy)
        w = x2 - x1
        h = y2 - y1
        padding_w, padding_h = int(0.1 * w), int(0.1 * h)
        x1 = max(x1 - padding_w, 0)
        y1 = max(y1 - padding_h, 0)
        x2 = min(x2 + padding_w, sx)
        y2 = min(y2 + padding_h, sy)

        cropped = image[y1:y2, x1:x2, :]
        self.cropped = cropped

        preprocessed_image = self._preprocessing_pipeline.preprocess_image(cropped)
        preprocessed_image = preprocessed_image.unsqueeze(0)

        return preprocessed_image, (x1, y1), (w + 2 * padding_w, h + 2 * padding_h)

    def _postprocess_model_output(
        self,
        out: torch.Tensor,
        preprocessed_image_shape: t.Tuple[int, int],
        top_left_face_pointsce_points: t.Tuple[int, int],
    ) -> np.ndarray:
        out = out.squeeze()

        w, h = preprocessed_image_shape

        out = torch.mul(out, torch.tensor([w, h]))

        out = torch.add(out, torch.tensor(top_left_face_pointsce_points))

        out = out.numpy().astype(np.uint32)

        return out

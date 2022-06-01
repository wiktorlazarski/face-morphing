from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class KeyPointsModel(nn.Module):
    def __init__(
        self,
        number_of_cordinates: int = 388,
        base_model: str = "resnet50",
        pretrained: bool = True,
    ):
        super().__init__()
        self.number_of_cordinates = number_of_cordinates
        self.number_of_points = number_of_cordinates // 2
        self.model = timm.create_model(base_model, pretrained=pretrained)
        in_channels = self.model.get_classifier().in_features
        self.model.fc = nn.Linear(
            in_features=in_channels, out_features=number_of_cordinates, bias=True
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.reshape(
            F.sigmoid(self.model(batch)), (batch.shape[0], self.number_of_points, 2)
        )

    @staticmethod
    def load_from_checkpoint(ckpt_path: str) -> KeyPointsModel:
        ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))

        hparams = ckpt["hyper_parameters"]
        neural_net = KeyPointsModel(
            number_of_cordinates=hparams["number_of_cordinates"],
            base_model=hparams["base_model"],
            pretrained=False,
        )

        weigths = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()}
        neural_net.load_state_dict(weigths, strict=True)

        return neural_net

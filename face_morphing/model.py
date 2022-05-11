from __future__ import annotations

import typing as t

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class KeyPointsModel(nn.Module):
    def __init__(self, number_of_cordinates: int = 388, base_model: str = 'resnet50'):
        super().__init__()
        self.number_of_cordinates = number_of_cordinates
        self.number_of_points = number_of_cordinates // 2
        self.model = timm.create_model(base_model, pretrained=True)
        in_channels = self.model.get_classifier().in_features
        self.model.fc = nn.Linear(in_features=in_channels, out_features=number_of_cordinates, bias=True)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.reshape(self.model(batch), (batch.shape[0], self.number_of_points, 2))


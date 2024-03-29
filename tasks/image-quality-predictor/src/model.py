from typing import Any

from torchvision import models

import torch.nn as nn


class CNNModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.features = models.resnet18(pretrained = True)
        self.features.fc = nn.Linear(512, 1)

        # Adding sigmoid activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Any) -> Any:
        x = self.features(x)
        x = self.sigmoid(x)
        return x.view(-1)

import torch
import torch.nn as nn
from torchvision import models


class CNNRegressor(nn.Module):
    def __init__(self, backbone: str = "resnet18", num_targets: int = 5):
        super().__init__()

        if backbone == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif backbone == "resnet34":
            m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif backbone == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        in_features = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m

        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)

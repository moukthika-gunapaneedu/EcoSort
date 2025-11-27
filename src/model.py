import torch
import torch.nn as nn
from torchvision import models

class EcoSortCNN(nn.Module):
    def __init__(self, num_classes: int = 6):
        super(EcoSortCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # 224x224 -> 14x14
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class EcoSortResNet18(nn.Module):
    """Transfer-learning model for better accuracy (demo version)."""
    def __init__(self, num_classes: int = 6, freeze_backbone: bool = True):
        super(EcoSortResNet18, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def build_model(num_classes: int = 6,
                device: str | None = None,
                model_type: str = "cnn"):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "resnet18":
        model = EcoSortResNet18(num_classes=num_classes, freeze_backbone=True)
    else:
        model = EcoSortCNN(num_classes=num_classes)

    model.to(device)
    return model, device

if __name__ == "__main__":
    m, d = build_model(model_type="cnn")
    x = torch.randn(4, 3, 224, 224).to(d)
    y = m(x)
    print("CNN output:", y.shape)
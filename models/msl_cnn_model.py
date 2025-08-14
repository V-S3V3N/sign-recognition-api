import torch
import torch.nn as nn

class MY3DCNN(nn.Module):
    def __init__(self, num_classes=44):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(64, 128, 3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def extract_features(self, x):
        with torch.no_grad():
            x = self.features(x)           # -> [B, 128, 1, 1, 1]
            x = x.view(x.size(0), -1)      # -> [B, 128]
        return x.squeeze(0)                # -> [128] if B=1
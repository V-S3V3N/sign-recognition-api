import torch.nn as nn

class FusionMLP(nn.Module):
    def __init__(self, input_dim=803, hidden_dim=256, num_classes=44):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)
import torch
from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()

        self.Conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.Linear(256 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Dropout(0.4),
        )

    def forward(self, x):
        x = self.Conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.output(x)
        return x


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters())}"
    )

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

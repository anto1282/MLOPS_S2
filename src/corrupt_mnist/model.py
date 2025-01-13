import torch
from torch import nn
import hydra

class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self, image_size, first_channels) -> None:
        super().__init__()
        final_image_size = image_size/4
        
        self.Conv = nn.Sequential(
            nn.Conv2d(1, first_channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(first_channels, first_channels*2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(first_channels*2, first_channels*4, 3, padding=1),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.Linear(first_channels*4 * final_image_size * final_image_size, 128),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(128, 10),
            nn.Dropout(0.4))

    def forward(self, x):
        x = self.Conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.output(x)
        x = self.fc(x)
        return x
    
@hydra.main("configs/model.yaml")
def main(cfg):
    
    image_size = cfg.hyperparameters.image_size
    first_channels = cfg.hyperparameters.first_channels
    
    model = MyAwesomeModel(image_size,first_channels)
    
    print(f"Model architecture: {model}")
    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters())}"
    )

    dummy_input = torch.randn(1, 1, image_size, image_size)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    return model
    

if __name__ == "__main__":
    model = main()
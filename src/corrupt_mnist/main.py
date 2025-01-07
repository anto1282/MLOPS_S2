import torch
import typer
from data import corrupt_mnist
from corrupt_mnist.src.corrupt_mnist.model import MyAwesomeModel
from torch import optim
from tqdm import tqdm
from torch import nn
import helper
import torch.nn.functional as F
import matplotlib.pyplot as plt

app = typer.Typer()

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)




@app.command()
def evaluate(model_checkpoint: str = "My_model.pt") -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_dataloader = corrupt_mnist(64)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img = img.unsqueeze(1)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    typer.run(train)

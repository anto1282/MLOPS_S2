import torch
import typer
from data import corrupt_mnist
from model import MyAwesomeModel

app = typer.Typer()
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


@app.command()
def evaluate(model_checkpoint: str = "models/My_model.pt") -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_dataloader = corrupt_mnist(64)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    typer.run(evaluate)

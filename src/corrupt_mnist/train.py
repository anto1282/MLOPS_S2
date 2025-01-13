
import torch
import typer
import os
from data import corrupt_mnist
from model import MyAwesomeModel
from torch import optim
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import hydra

app = typer.Typer()

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


@hydra.main(config_name="train.yaml")
def train(cfg) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    lr = cfg.hyperparameters.lr
    batch_size = cfg.hyperparameters.batch_size
    epochs = cfg.hyperparameters.epochs
    
    print(f"{lr=}, {batch_size=}, {epochs=}")

    model = MyAwesomeModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    trainloader, _ = corrupt_mnist(batch_size)
    statistics = {"train_loss": [], "train_accuracy": []}

    for epoch in range(epochs):
        model.train()
        for i, batch in tqdm(enumerate(trainloader)):
            images, labels = batch
            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (outputs.argmax(dim=1) == labels).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

        print(f"Loss at epoch {epoch+1}: {loss}")

    print("Training complete")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig(
        f"reports/figures/training_statistics_lr_{lr}_epochs_{epochs}.png"
    )
    torch.save(
        model.state_dict(), f"models/My_model_lr_{lr}_epochs_{epochs}.pt"
    )
if __name__ == "__main__":
    typer.run(train)

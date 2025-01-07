import torch
import typer
from data import corrupt_mnist
from model import MyAwesomeModel
from torch import optim
from tqdm import tqdm
from torch import nn
import helper
import torch.nn.functional as F 
import matplotlib.pyplot as plt
app = typer.Typer()


@app.command()
def train(lr: float = 1e-3, epochs: int = 10) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    batch_size = 64
    trainloader, _ = corrupt_mnist(batch_size)
    statistics = {"train_loss": [], "train_accuracy": []}
    
    for epoch in range(epochs): 
        model.train()   
        for i,batch in tqdm(enumerate(trainloader)):
            
            images,labels = batch            
            images = images.unsqueeze(1)
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
    fig.savefig("training_statistics.png")
    torch.save(model.state_dict(),"My_model.pt")

@app.command()
def evaluate(model_checkpoint: str = "My_model.pt")  -> None:
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
    app()
import torch
import typer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


class corruptMnistDataset(Dataset):
    def __init__(self, images, labels):
        super().__init__()
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        image_tensor = self.images[index]
        label = self.labels[index]

        return image_tensor, label

    def __len__(self):
        return len(self.images)


def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    train_images = torch.stack(
        [torch.load(raw_dir + f"/train_images_{x}.pt", weights_only=False) for x in range(0, 6)],
        dim=0,
    ).flatten(end_dim=1)
    train_targets = torch.stack(
        [torch.load(raw_dir + f"/train_target_{x}.pt", weights_only=False) for x in range(0, 6)],
        dim=0,
    ).flatten(end_dim=1)

    test_images = torch.load(raw_dir + "/test_images.pt", weights_only=False)
    test_targets = torch.load(raw_dir + "/test_target.pt", weights_only=False)

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()

    train_targets = train_targets.long()
    test_targets = test_targets.long()

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_targets, f"{processed_dir}/train_target.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(test_targets, f"{processed_dir}/test_target.pt")


def corrupt_mnist(batchsize=64):
    train_images = torch.load("data/processed/train_images.pt", weights_only=False)
    train_targets = torch.load("data/processed/train_target.pt", weights_only=False)
    test_images = torch.load("data/processed/test_images.pt", weights_only=False)
    test_targets = torch.load("data/processed/test_target.pt", weights_only=False)

    train_set = corruptMnistDataset(train_images, train_targets)
    test_set = corruptMnistDataset(test_images, test_targets)

    trainLoader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
    testLoader = DataLoader(test_set, batch_size=batchsize, shuffle=False)

    return trainLoader, testLoader


if __name__ == "__main__":
    typer.run(preprocess_data)

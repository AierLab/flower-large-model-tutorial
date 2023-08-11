import torch
from torchvision import datasets, transforms


def get_dataloader(train=True, batch_size=32):
    """
    Fetches the MNIST data loader.

    Args:
        train (bool): If True, returns the training data loader. Otherwise, returns the test data loader.
        batch_size (int): Batch size for the dataloader.

    Returns:
        DataLoader: Torch data loader for the MNIST dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST('./data', train=train,
                             download=True, transform=transform)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

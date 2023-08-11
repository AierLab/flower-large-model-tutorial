import torch
import torch.nn as nn
import torch.nn.functional as F


class DemoModel(nn.Module):
    """
    Simple Feed-Forward Neural Network for demonstration purposes.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        model_type (str): A string to denote if it's a client or server model.
    """

    def __init__(self, model_type):
        super(DemoModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
        self.model_type = model_type

    def forward(self, x):
        """Define the forward pass for the model."""
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save_local(self, epoch, loss):
        """
        Saves the model locally with its state.

        Args:
            epoch (int): The epoch number.
            loss (float): The loss value.
        """
        state = {
            'epoch': epoch,
            'state_dict': self.state_dict(),
            'loss': loss,
        }
        torch.save(state, f"model_epoch_{epoch}.pth")

    @staticmethod
    def load_local(filepath):
        """
        Loads the model and its state from a local file.

        Args:
            filepath (str): Path to the model file.

        Returns:
            tuple: Loaded model, last epoch number, and last loss value.
        """
        model = DemoModel("client")
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, epoch, loss

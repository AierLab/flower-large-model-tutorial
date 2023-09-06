import os
import time
from typing import Tuple
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

class AbstractModel(nn.Module, ABC):
    
    def __init__(self, model_folder, fed_model_dir):
        """
        Initialize the AbstractModel class.
        Args:
            model_folder (str): Directory where the local model parameters are stored.
            fed_model_dir (str): Directory where the federated model parameters are stored.
        """
        super(AbstractModel, self).__init__()
        self.model_folder = model_folder
        self.fed_model_dir = fed_model_dir

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def model_train(self, dataloader: DataLoader, epochs: int, device: torch.device):
        pass

    @abstractmethod
    def model_test(self, dataloader: DataLoader, device: torch.device = None) -> Tuple[float, float]:
        pass

    def get_latest_model_file(self, directory):
        model_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        return max(model_files, key=os.path.getctime) if model_files else ""

    def save_local(self, epoch: int, loss, optimizer_state_dict: dict) -> None:
        """
        Save the model locally.
        """
        filename = f'model_epoch_{epoch}_{int(time.time())}.pt'
        path = os.path.join(self.model_folder, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer_state_dict,
            'loss': loss,
        }, path)
        print(f"Log: Model saved at {path}")

    def load_local(self) -> dict:
        """
        Load the latest model from either the local or federated directories, whichever is most recent.
        """
        latest_local_model_file = self.get_latest_model_file(self.model_folder)
        latest_fed_model_file = self.get_latest_model_file(self.fed_model_dir)

        if latest_local_model_file and latest_fed_model_file:
            if os.path.getctime(latest_local_model_file) > os.path.getctime(latest_fed_model_file):
                print(f"Log: Loading latest local model from {latest_local_model_file}")
                return torch.load(latest_local_model_file)
            else:
                print(f"Log: Loading latest federated model from {latest_fed_model_file}")
                return torch.load(latest_fed_model_file)
        elif latest_local_model_file:
            print(f"Log: Loading latest local model from {latest_local_model_file}")
            return torch.load(latest_local_model_file)
        elif latest_fed_model_file:
            print(f"Log: Loading latest federated model from {latest_fed_model_file}")
            return torch.load(latest_fed_model_file)
        else:
            print("Log: No local or federated model found.")
            return None  # or initialize a new model here, if necessary



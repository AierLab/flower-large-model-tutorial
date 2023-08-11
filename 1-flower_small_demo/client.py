import flwr as fl
import torch
import torch.optim as optim
from model import DemoModel
from data import get_dataloader
import glob

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_weights(model):
    """Retrieve weights from a PyTorch model."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, weights):
    """Set weights for a PyTorch model."""
    model_dict = model.state_dict()
    params_keys = list(model_dict.keys())
    for key, val in zip(params_keys, weights):
        model_dict[key] = torch.Tensor(val).to(DEVICE)
    model.load_state_dict(model_dict, strict=True)


class FedClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = DemoModel("client").to(DEVICE)
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.5)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loader = get_dataloader(train=True, batch_size=32)
        self.test_loader = get_dataloader(train=False, batch_size=32)

        # Load the latest model if available
        latest_model_file = self.get_latest_model_file()
        if latest_model_file:
            self.model, self.last_epoch, _ = DemoModel.load_local(
                latest_model_file)
        else:
            self.last_epoch = 0

    @staticmethod
    def get_latest_model_file():
        """Returns the filepath of the latest saved model based on epoch number."""
        model_files = sorted(glob.glob("model_epoch_*.pth"))
        return model_files[-1] if model_files else None

    def get_parameters(self):
        return get_weights(self.model)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)

        for epoch in range(self.last_epoch, self.last_epoch + 2):  # Train for 2 epochs
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                self.optimizer.zero_grad()
                output = self.model(data.view(data.size(0), -1))
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                # For demonstration purposes, let's print every 100 batches
                if batch_idx % 100 == 0:
                    print(
                        f"Epoch: {epoch} Batch: {batch_idx} Loss: {loss.item():.4f}")

            # Saving after every epoch
            self.model.save_local(epoch, loss.item())

        return get_weights(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = self.model(data.view(data.size(0), -1))
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100. * correct / len(self.test_loader.dataset)
        return test_loss, len(self.test_loader.dataset)


if __name__ == "__main__":
    fl.client.start_numpy_client("localhost:8080", client=FedClient())

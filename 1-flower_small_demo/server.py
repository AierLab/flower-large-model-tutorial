import flwr as fl
from model import DemoModel
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def start_server():
    """Initialize and start the Flower server."""
    model = DemoModel("server").to(DEVICE)
    strategy = fl.server.strategy.FedAvg()
    server = fl.server.Server(
        model=model, config={"num_rounds": 3}, strategy=strategy)
    server.fit()


if __name__ == "__main__":
    start_server()

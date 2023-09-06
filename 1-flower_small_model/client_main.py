from data import CifarData
from fedlearn import FedClient
from model import DemoModel
import argparse

# Define the main function
if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add an argument for the server address
    parser.add_argument(
        "-addr",
        type=str,
        default="0.0.0.0:8080",
        help="Server address. Defaults to \"0.0.0.0: 8080\".",
    )

    # Add an argument for the client rank
    parser.add_argument(
        "-rank",
        type=int,
        default=1,
        help="Client rank, the id of client.",
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Extract the address and rank from the arguments
    addr = args.addr
    rank = int(args.rank)

    # Define the client directory
    CLIENT_DIR = f"../tmp/client/c{str(rank).zfill(2)}"

    # Initialize the data and model
    data = CifarData(data_dir=CLIENT_DIR)
    model = DemoModel(model_dir=CLIENT_DIR)

    # Create a FedClient instance
    client = FedClient(data, model)

    # Run the client
    client.run(addr)


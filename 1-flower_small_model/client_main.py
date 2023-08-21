from data import CifarData
from fedlearn import FedClient
from model import DemoModel
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-addr",
        type=str,
        default="0.0.0.0:8080",
        help="Server address. Defaults to \"0.0.0.0: 8080\".",
    )

    parser.add_argument(
        "-rank",
        type=int,
        default=1,
        help="Client rank, the id of client.",
    )

    args = parser.parse_args()

    addr = args.addr
    rank = int(args.rank)

    CLIENT_DIR = f"../tmp/client/c{str(rank).zfill(2)}"

    # Init data and model.
    data = CifarData(data_dir=CLIENT_DIR)
    model = DemoModel(model_dir=CLIENT_DIR)

    client = FedClient(data, model)
    client.run(addr)

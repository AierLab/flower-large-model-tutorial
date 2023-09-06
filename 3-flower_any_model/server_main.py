import argparse

# Importing the necessary modules
from model import DemoModel
from fedlearn import FedServer

# Main function
if __name__ == '__main__':
    # Start Flower server for three rounds of federated learning
    # Creating an ArgumentParser object
    parser = argparse.ArgumentParser()
    
    # Adding arguments for the number of rounds for the federated training
    parser.add_argument(
        "-r", type=int, default=3, help="Number of rounds for the federated training"
    )

    # Adding arguments for the batch size
    parser.add_argument("-b", type=int, default=32, help="Batch size")

    # Adding arguments for the minimum number of clients to be sampled next round
    parser.add_argument(
        "-fc",
        type=int,
        default=2,
        help="Min fit clients, min number of clients to be sampled next round",
    )

    # Adding arguments for the minimum number of clients that need to connect to the server before training round can start
    parser.add_argument(
        "-ac",
        type=int,
        default=2,
        help="Min available clients, min number of clients that need to connect to the server before training round can start",
    )

    # Adding arguments for the minimum number of clients to be sampled for evaluation
    parser.add_argument(
        "-ec",
        type=int,
        default=2,
        help="Min evaluate clients, min number of clients to be sampled for evaluation",
    )

    # Adding arguments for the path to checkpoint to be loaded
    parser.add_argument(
        "-ckpt",
        type=str,
        default="",
        help="Path to checkpoint to be loaded",
    )

    # Adding arguments for the server address
    parser.add_argument(
        "-addr",
        type=str,
        default="0.0.0.0:8080",
        help="Server address. Defaults to \"0.0.0.0: 8080\".",
    )

    # Parsing the arguments
    args = parser.parse_args()

    # Assigning the parsed arguments to variables
    rounds = int(args.r)
    fc = int(args.fc)
    ac = int(args.ac)
    ec = int(args.ec)
    ckpt_path = args.ckpt
    addr = args.addr
    global batch_size
    batch_size = int(args.b)
    init_param = None

    # Defining the server directory
    SERVER_DIR = "../tmp/server"
    
    # Creating a DemoModel object
    model = DemoModel(SERVER_DIR, SERVER_DIR)
    base_epoch = 0

    # Loading the model
    model_dict = model.load_local()
    if model_dict:
        model.load_state_dict(model_dict["model_state_dict"])
        base_epoch = model_dict["epoch"]

    # Creating a FedServer object
    server = FedServer(model, base_epoch, fc, ac)
    
    # Running the server
    server.run(addr)

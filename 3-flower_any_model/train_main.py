import argparse
import os
import torch
from data import CifarData
from model import DemoModel  # Please replace with your actual model class that inherits from AbstractModel
from torch.utils.data import DataLoader  # Please replace with your actual data loading mechanism

def main():    
    # Create an argument parser
    parser = argparse.ArgumentParser()

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
    rank = int(args.rank)
    
    # Define the client directory
    CLIENT_DIR = f"../tmp/client/c{str(rank).zfill(2)}"
    
    # Define the model_folder and fed_model_dir paths
    model_folder = os.path.join(CLIENT_DIR, "train")
    fed_model_dir = os.path.join(CLIENT_DIR, "fed")
    
    
    # Create a data loader instance (replace with your actual data loader)
    data = CifarData(data_dir=CLIENT_DIR)

    
    # Create a device instance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an instance of your model (replace `YourConcreteModel` with your actual model class name)
    model = DemoModel(model_folder=model_folder, fed_model_dir=fed_model_dir)

    # Define the number of epochs
    epochs = 10  # Adjust as necessary

    # Train the model
    model.model_train(data.trainloader, epochs, device)

    # Save the model locally
    # You would need to have logic inside your model_train method to save the model at the end of training
    
    loss, accuracy = model.model_test(data.testloader, device)
    print(f"loss: {loss}, accuracy{accuracy}")

if __name__ == "__main__":
    main()

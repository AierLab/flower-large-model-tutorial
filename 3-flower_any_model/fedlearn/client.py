import os
import time
import torch
from collections import OrderedDict
import flwr as fl

class FedClient(fl.client.NumPyClient):
    """
    FlowerClient class for federated learning. This class implements the methods required for federated learning,
    where each client owns a portion of the total data and trains a model on its own data.
    """
    def __init__(self, model_folder, fed_model_dir):
        """
        Initialize the FlowerClient.
        Args:
            model_folder (str): Directory where the local model parameters are stored.
            fed_model_dir (str): Directory where the federated model parameters are stored.
        """
        self.model_folder = model_folder
        self.fed_model_dir = fed_model_dir
        self.latest_model_file = self.get_latest_model_file()
        self.latest_fed_model_file = ""

    def get_latest_model_file(self):
        model_files = [os.path.join(self.model_folder, f) for f in os.listdir(self.model_folder) if os.path.isfile(os.path.join(self.model_folder, f))]
        return max(model_files, key=os.path.getctime) if model_files else ""

    def get_parameters(self):
        """
        Get the parameters of the model.
        Returns:
            parameters (list): List of the model parameters.
        """
        if self.latest_model_file:
            checkpoint = torch.load(self.latest_model_file)
            state_dict = checkpoint['model_state_dict']  # Get only the state_dict of the model
            parameters = [val.cpu().numpy() for _, val in state_dict.items() if torch.is_tensor(val)]
            return parameters
        else:
            raise FileNotFoundError("No model files found in the model folder.")


    def set_parameters(self, parameters):
        """
        Set the parameters of the model.
        Args:
            parameters (list): List of the model parameters.
        """
        if self.latest_model_file:
            # Load the existing state_dict from the latest model file
            checkpoint = torch.load(self.latest_model_file)
            state_dict = checkpoint['model_state_dict']
            
            # Update the state_dict with the new parameters
            for param, (key, value) in zip(parameters, state_dict.items()):
                state_dict[key] = torch.Tensor(param)
            
            # Generate a unique filename based on the current time
            self.latest_fed_model_file = os.path.join(self.fed_model_dir, f"fed_model_{time.time()}.pt")
            
            # Update the checkpoint with the new state_dict
            checkpoint['model_state_dict'] = state_dict
            
            # Save the updated checkpoint to the file
            torch.save(checkpoint, self.latest_fed_model_file)
        else:
            raise FileNotFoundError("No model files found in the model folder.")


    def fit(self, parameters, config):
        """
        Fit the model on the client's data.
        Args:
            parameters (list): List of the model parameters.
            config (dict): Configuration dictionary.
        Returns:
            parameters (list): List of the updated model parameters.
            len(self.trainloader) (int): Length of the training data loader.
            {} (dict): Empty dictionary.
        """
        # self.set_parameters(parameters)
        while self.latest_model_file == self.get_latest_model_file():
            time.sleep(0.1)
        self.latest_model_file = self.get_latest_model_file()
        
        # Assuming self.trainloader is defined elsewhere in your program
        # Here we're only returning the latest parameters without further training
        return self.get_parameters(), 500, {} # TODO change this 1000 to the number of training data. or batch size for fed learning.

    def evaluate(self, parameters, config):
        """
        Evaluate the model on the client's data.
        Args:
            parameters (list): List of the model parameters.
            config (dict): Configuration dictionary.
        Returns:
            float(loss) (float): Loss of the model.
            len(self.valloader) (int): Length of the validation data loader.
            {'accuracy': accuracy} (dict): Dictionary with the accuracy of the model.
        """
        self.set_parameters(parameters)
        
        # Here, you'd perform your evaluation using self.valloader, calculate loss and accuracy
        # Assuming loss and accuracy are computed elsewhere in your code.
        # If not, you'd need to integrate your evaluation logic here to calculate loss and accuracy.
        
        # Placeholder values, replace with actual loss and accuracy computation
        loss = 0.0  
        accuracy = 0.0  # TODO update these values, also call evaluation function from the model.

        # Assuming self.valloader is defined elsewhere in your program
        return float(loss), 500, {'accuracy': accuracy} # TODO change this 1000 to the number of training data. or batch size for fed learning.
    
    def run(self, server_address):
        fl.client.start_numpy_client(
            server_address=server_address, client=self)

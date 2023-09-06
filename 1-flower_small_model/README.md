# Repository Overview

This repository has been updated to include a variety of machine learning models and techniques, with a focus on federated learning. The models are implemented using the PyTorch framework and the Flower library for federated learning.

## Environment Setup
1. Install the Flower framework.
  ```
  python -m pip install flwr 
  ```
2. Install Pytorch. It is recommended to follow the official guidance for installation. 
  ```
  python -m pip install torch torchvision
  ```

## Running the Models
To run the models, navigate to the appropriate folder and start the server.
  ```
  cd path/to/the/folder
  python server_main.py
  ```
Then, in a new terminal, start the first client.
  ```
  cd path/to/the/folder
  python client_main.py -rank 1
  ```
Repeat the previous step in a third terminal to start the second client. 
  ```
  cd path/to/the/folder
  python client_main.py -rank 2
  ```
You can add as many clients as you want. A normal simulation requires at least two clients.

> Configuration options can be viewed with the `-h` or `--help` option.

## Features

The repository includes the following features:
- Federated learning implemented with PyTorch.
- Ability to load and save checkpoints on both the server and client sides.
- Customizable aggregation function for split learning.

Please note that the repository is under active development, and more features will be added in the future.

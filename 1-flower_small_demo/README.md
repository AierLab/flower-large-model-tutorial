# Federated Learning Project

## Requirements:
- Python 3.7+
- Libraries: torch, torchvision, flwr

## Installation:
```bash
pip install torch torchvision flwr
```

## Running the Project:

1. **Start the server**: 
```bash
$ python server.py
```

2. **Start the client(s)**: 
```bash
$ python client.py
```

## Description:
The server uses federated learning to train a model with the help of clients. Each client has a subset of the dataset, and the model is trained collaboratively without exposing the data of individual clients.

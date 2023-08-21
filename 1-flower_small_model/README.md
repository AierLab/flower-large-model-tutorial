# Federated Learning

Federated learning with model weights aggregation, with updated aggregation, able to save model weights.

## Environment Setup
1.Install [Flower](https://flower.dev/docs/install-flower.html#install-stable-release) framework.
  ```
  python -m pip install flwr 
  ```
2.Install Pytorch, better following the official guidence. 
  ```
  python -m pip install torch torchvision
  ```
## Run Federated Learning
  ```
  cd path/to/this/folder
  ```
  Start the server first.
  ```
  python server_main.py
  ```
  Then open a new terminal to start the first client.
  ```
  cd path/to/this/folder
  python client_main.py -rank 1
  ```
  Repeat the previous step in the third terminal to start the second client. 
  
  ```
  cd path/to/this/folder
  python client_main.py -rank 2
  ```

  Normal simulation requires at least two clients, you can add as many clients as you want.
  
  
> You can view the configuration options with following option: `-h` or `--help`.

## Implemented Features

- [x] Federated learning + Pytroch.
- [x] Load checkpoint on server side.
- [x] Save checkpoint on client side.
- [x] Save checkpoint on server side.
- [x] Load checkpoint on client side.
- [ ] Rewrite Aggregation func for split learning.

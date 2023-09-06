# Federated Learning with Independent Training Thread

Welcome to our groundbreaking Federated Learning repository where we redefine the paradigms of distributed machine learning. By segregating the training process from the core Flower framework and leveraging local state dictionaries as the medium for intra-system communication, we've crafted a setup where complex training methodologies can seamlessly integrate without impinging on the native Flower framework logic.

## Setting Up Your Development Environment

Before you venture into the repository, ensure your system is equipped with the necessary frameworks to facilitate a smooth experience:

1. **Flower Framework Installation:**
   
   Set up the Flower framework, a foundational element for federated learning. Execute the following command in your terminal:
   
   ```shell
   python -m pip install flwr 
   ```

2. **PyTorch Installation:**
   
   Subsequently, install PyTorch, a powerful library facilitating machine learning and deep learning implementations. Utilize the official guidelines for a hassle-free installation:
   
   ```shell
   python -m pip install torch torchvision
   ```

## Running the Simulation

Our approach separates the model training aspect from the Flower framework, promoting more complex training methodologies without any disruptions. Here's your roadmap to executing the simulation:

1. **Server Startup:**
   
   Traverse to the designated project directory and initialize the server with the following command:
   
   ```shell
   cd path/to/the/folder
   python server_main.py
   ```

2. **Client Main Activation:**
   
   In separate terminals, kickstart the client main instances sequentially with assigned ranks:
   
   ```shell
   cd path/to/the/folder
   python client_main.py --rank 1
   ```
   ```shell
   cd path/to/the/folder
   python client_main.py --rank 2
   ```

3. **Model Main Activation:**
   
   In parallel, launch the model training threads in distinct terminals, providing them with respective ranks:
   
   ```shell
   cd path/to/the/folder
   python model_main.py --rank 1
   ```
   ```shell
   cd path/to/the/folder
   python model_main.py --rank 2
   ```

This establishes a standard simulation setup, orchestrating five concurrent terminals: two hosting `client_main.py`, two accommodating `model_main.py`, and one dedicated to `server_main.py`.

> Explore the variety of configuration options with the `-h` or `--help` command-line arguments.

## Features and Advancements

Immerse yourself in our enriched repository that offers:

- **Detached Training Process:** Unlinking the training process from the Flower framework, opening avenues for intricate training schemes to function harmoniously without interfering with the Flower framework's core logic.
- **Localized State Dictionary Communication:** Utilizing local state dictionaries as a mode of internal communication, fostering a landscape where more sophisticated training techniques can be employed with ease and efficiency.
- **Checkpoint Management:** Facilitating the management of checkpoints both server-side and client-side, instilling a resilient learning protocol.
- **Adaptable Aggregation Functionality:** Customize the aggregation functions to cater to your split learning requirements, facilitating an optimized learning workflow crafted for specific project requisites.

As we delve deeper into enhancing our repository, anticipate new, dynamic features to be integrated soon.

Join us in this endeavor as we aim to redefine federated learning protocols, fostering an environment where complexity meets simplicity, paving the way for the next generation of machine learning solutions. Thank you for being a part of this revolutionary journey!
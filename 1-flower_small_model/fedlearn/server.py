import argparse
from typing import List, Union, Optional, Tuple, Dict

import flwr as fl
import numpy as np
from flwr.common import FitRes, Scalar, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from model import AbstractModel, DemoModel
from helper import get_weights, set_weights


class FedAvgSaveModel(fl.server.strategy.FedAvg):
    def __init__(self, model: AbstractModel, base_epoch: int, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.base_epoch = base_epoch

    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
                      ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        server_round = self.base_epoch + server_round
        weights = super().aggregate_fit(server_round, results, failures)
        if weights[0] is not None:
            # Save weights
            print(f"Log: Saving round {server_round} weights...")
            set_weights(self.model, parameters_to_ndarrays(weights[0]))
            # TODO update to meaningful info
            self.model.save_local(server_round, None, None)
        return weights


class FedServer:
    def __init__(self, model: AbstractModel, base_epoch: int, fc, ac):
        # get weights of model
        init_weights = get_weights(model)
        # Convert the weights (np.ndarray) to parameters
        init_param = fl.common.ndarrays_to_parameters(init_weights)

        self.strategy = FedAvgSaveModel(
            model=model,
            base_epoch=base_epoch,
            fraction_fit=1.0,  # Sample 100% of available clients for training
            # fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
            min_fit_clients=fc,  # Never sample less than 2 clients for training
            # min_evaluate_clients=ec,  # Never sample less than 2 clients for evaluation
            min_available_clients=ac,  # Wait until all 2 clients are available
            initial_parameters=init_param,
        )

    def run(self, server_address):
        fl.server.start_server(server_address=server_address,
                               config=fl.server.ServerConfig(num_rounds=3),
                               strategy=self.strategy)

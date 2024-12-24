import time
from typing import List, Tuple

from flwr.common import (
    Context,
    FitIns,
    Metrics,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate_inplace

from src.ml_models.net import Net
from src.ml_models.utils import get_weights, set_weights


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def combine_weights(global_weights, degenerated_weights, lambda_val=0.95):
    """Combine global and degenerated model weights."""
    return [
        lambda_val * g + (1 - lambda_val) * d
        for g, d in zip(global_weights, degenerated_weights)
    ]


class UnlearningFedAvg(FedAvg):
    def __init__(self, num_of_clients, **kwargs):
        super().__init__(**kwargs)
        self.unlearning_initiated_by_client_id = -1
        self.degenerated_model = None
        self.num_of_clients = num_of_clients

    def configure_fit(self, server_round, parameters, client_manager):
        # Waiting till all clients are connected
        client_manager.wait_for(self.num_of_clients)

        # Calling the parent class's configure_fit method
        client_fit_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        # Modifying the config for each FitIns object
        for _, fit_ins in client_fit_pairs:
            # Add the current round to the config
            fit_ins.config["current_round"] = server_round
            fit_ins.config["unlearning_initiated_by_client_id"] = (
                self.unlearning_initiated_by_client_id
            )
            print("fit_ins.config", fit_ins.config)

        return client_fit_pairs

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        aggregated_ndarrays = aggregate_inplace(results)

        for _, fit_res in results:
            if fit_res.metrics.get("unlearning_initiated_by_client_id", -1) != -1:
                self.unlearning_initiated_by_client_id = fit_res.metrics.get(
                    "unlearning_initiated_by_client_id"
                )

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}

        return parameters_aggregated, metrics_aggregated


def server_fn(context: Context):
    print("context.node_config", context)
    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = UnlearningFedAvg(
        fraction_evaluate=float(context.run_config["fraction-evaluate"]),
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
        num_of_clients=int(context.run_config["num-of-clients"]),
    )
    config = ServerConfig(num_rounds=int(context.run_config["num-server-rounds"]))

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)

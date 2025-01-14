from functools import reduce
from typing import List, Tuple

import numpy as np
from flwr.common import (
    Context,
    Metrics,
    NDArrays,
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


def custom_aggregate(results: list[tuple[NDArrays, float]]) -> NDArrays:
    """
    Aggregate model parameters with custom weightages.

    Parameters:
    results: List of tuples, where each tuple contains:
        - NDArrays: Model parameters
        - float: Weightage for this model (e.g., 0.1 for 10%, 0.9 for 90%)

    Returns:
    NDArrays: Aggregated model parameters.
    """
    # Ensure weightages sum up to 1 for valid aggregation
    total_weight = sum(weight for _, weight in results)
    if not np.isclose(total_weight, 1.0):
        raise ValueError("Weightages must sum up to 1.0")

    # Multiply model weights by their respective weightage
    weighted_weights = [
        [layer * weight for layer in weights] for weights, weight in results
    ]

    # Sum up the weighted layers across models
    aggregated_weights: NDArrays = [
        reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
    ]

    return aggregated_weights


class UnlearningFedAvg(FedAvg):
    def __init__(self, num_of_clients, **kwargs):
        super().__init__(**kwargs)
        self.unlearning_initiated_by_client_id = -1
        self.command = ""
        self.next_command = ""
        self.num_of_clients = num_of_clients

        self.degraded_model_parameters = None
        self.global_model_parameters = None

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
            fit_ins.config["command"] = self.command
            print("fit_ins.config", fit_ins.config)

        return client_fit_pairs

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        self.command = self.next_command

        filtered_results = []
        for client_proxy, fit_res in results:
            unlearning_id = fit_res.metrics.get("unlearning_initiated_by_client_id", -1)

            if unlearning_id != -1 and self.unlearning_initiated_by_client_id == -1:
                self.unlearning_initiated_by_client_id = unlearning_id
                self.command = "initialize_degraded_model_and_merge_with_SGA_model_of_target_client"

            if unlearning_id == self.unlearning_initiated_by_client_id:
                if (
                    self.command
                    == "initialize_degraded_model_and_merge_with_SGA_model_of_target_client"
                ):
                    initial_degraded_model_with_rand_parameters = get_weights(Net())
                    self.degraded_model_parameters = ndarrays_to_parameters(
                        custom_aggregate(
                            [
                                (fit_res.parameters, 0.5),
                                (initial_degraded_model_with_rand_parameters, 0.5),
                            ]
                        )
                    )
                    self.command = "perform_fl_on_remaining_clients_and_SGA_on_target_client_with_degraded_model"

            if (
                unlearning_id != self.unlearning_initiated_by_client_id
                or self.command
                == "merge_global_model_with_degraded_model_and_perform_global_model_restoration"
            ):
                filtered_results.append((client_proxy, fit_res))

        results = filtered_results

        aggregated_ndarrays = aggregate_inplace(results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}

        if (
            self.command
            == "perform_fl_on_remaining_clients_and_SGA_on_target_client_with_degraded_model"
        ):
            self.next_command = "merge_global_model_with_degraded_model_and_perform_global_model_restoration"

            self.global_model_parameters = parameters_aggregated

            return self.degraded_model_parameters, metrics_aggregated

        if (
            self.command
            == "merge_global_model_with_degraded_model_and_perform_global_model_restoration"
        ):
            self.next_command = "perform_fl_on_remaining_clients_and_SGA_on_target_client_with_degraded_model"
            self.global_model_parameters = ndarrays_to_parameters(
                custom_aggregate(
                    [
                        (parameters_aggregated, 0.9),
                        (self.degraded_model_parameters, 0.1),
                    ]
                )
            )
            self.degraded_model_parameters = parameters_aggregated

            return self.global_model_parameters, metrics_aggregated

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

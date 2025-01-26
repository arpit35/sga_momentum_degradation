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
    def __init__(
        self,
        num_of_clients,
        weight_factor_degradation_model,
        weight_factor_global_model,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.unlearn_client_number = -1
        self.unlearn_client_id = -1
        self.command = ""
        self.knowledge_eraser_rounds = 0

        self.num_of_clients = num_of_clients
        self.weight_factor_degradation_model = weight_factor_degradation_model
        self.weight_factor_global_model = weight_factor_global_model

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
        for client, fit_ins in client_fit_pairs:
            # Add the current round to the config
            fit_ins.config["current_round"] = server_round
            fit_ins.config["unlearn_client_number"] = self.unlearn_client_number
            fit_ins.config["command"] = self.command
            print("fit_ins.config", fit_ins.config)

            if (
                self.command == "global_model_restoration_and_degraded_model_unlearning"
                and self.unlearn_client_id == client.cid
            ):
                fit_ins.parameters = self.degraded_model_parameters

        return client_fit_pairs

    def configure_evaluate(self, server_round, parameters, client_manager):
        # Calling the parent class's configure_evaluate method
        client_evaluate_pairs = super().configure_evaluate(
            server_round, parameters, client_manager
        )

        for client, evaluate_ins in client_evaluate_pairs:
            # Add the current round to the config
            evaluate_ins.config["current_round"] = server_round
            evaluate_ins.config["unlearn_client_number"] = self.unlearn_client_number
            evaluate_ins.config["command"] = self.command

        return client_evaluate_pairs

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        filtered_results = []
        for client_proxy, fit_res in results:
            print("fit_res.metrics", fit_res.metrics)
            if fit_res.metrics.get("unlearn_client_number", -1) != -1:
                self.command = "degraded_model_initialization"

                self.unlearn_client_number = fit_res.metrics.get(
                    "unlearn_client_number"
                )

                self.unlearn_client_id = client_proxy.cid

                initial_degraded_model_with_rand_parameters = get_weights(Net())
                self.degraded_model_parameters = ndarrays_to_parameters(
                    custom_aggregate(
                        [
                            (
                                initial_degraded_model_with_rand_parameters,
                                self.weight_factor_degradation_model,
                            ),
                            (
                                parameters_to_ndarrays(fit_res.parameters),
                                1 - self.weight_factor_degradation_model,
                            ),
                        ]
                    )
                )
                continue

            if (
                self.command == "degraded_model_refinement"
            ) and self.unlearn_client_id == client_proxy.cid:
                continue

            if (
                self.command == "global_model_restoration_and_degraded_model_unlearning"
            ) and self.unlearn_client_id == client_proxy.cid:
                self.degraded_model_parameters = fit_res.parameters
                continue

            filtered_results.append((client_proxy, fit_res))

        parameters_aggregated = ndarrays_to_parameters(
            aggregate_inplace(filtered_results)
        )

        if self.command == "degraded_model_refinement":
            self.command = "global_model_unlearning"

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}

        if self.command == "global_model_unlearning":
            self.degraded_model_parameters = parameters_aggregated
            self.global_model_parameters = ndarrays_to_parameters(
                custom_aggregate(
                    [
                        (
                            parameters_to_ndarrays(self.global_model_parameters),
                            self.weight_factor_global_model,
                        ),
                        (
                            parameters_to_ndarrays(parameters_aggregated),
                            1 - self.weight_factor_global_model,
                        ),
                    ]
                )
            )
            return self.global_model_parameters, metrics_aggregated

        self.global_model_parameters = parameters_aggregated

        if self.command == "degraded_model_initialization":
            return self.degraded_model_parameters, metrics_aggregated

        if self.command == "global_model_restoration_and_degraded_model_unlearning":
            return self.degraded_model_parameters, metrics_aggregated

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(self, server_round, results, failures):

        if self.command == "degraded_model_initialization":
            self.command = "degraded_model_refinement"

        elif self.command == "global_model_unlearning":
            self.command = "global_model_restoration_and_degraded_model_unlearning"

        elif self.command == "global_model_restoration_and_degraded_model_unlearning":
            if self.knowledge_eraser_rounds == 4:
                self.unlearn_client_number = -1
                self.unlearn_client_id = -1
                self.command = ""
                self.knowledge_eraser_rounds = 0
            else:
                self.knowledge_eraser_rounds += 1
                self.command = "degraded_model_refinement"

        return super().aggregate_evaluate(server_round, results, failures)


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
        weight_factor_degradation_model=float(
            context.run_config["weight-factor-degradation-model"]
        ),
        weight_factor_global_model=float(
            context.run_config["weight-factor-global-model"]
        ),
    )
    config = ServerConfig(num_rounds=int(context.run_config["num-server-rounds"]))

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)

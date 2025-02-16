import json
import os
import pickle
from functools import reduce
from typing import List, Tuple

import numpy as np
from flwr.common import (
    Context,
    FitIns,
    Metrics,
    NDArrays,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate_inplace

from src.ml_models.net import get_net
from src.ml_models.utils import get_weights

# from src.utils.json_encoder import CustomEncoder


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
        mode,
        num_of_clients,
        num_server_rounds,
        weight_factor_degradation_model,
        weight_factor_global_model,
        dataset_num_channels,
        dataset_num_classes,
        model_name,
        dataset_name,
        plots_folder_path,
        models_folder_path,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.unlearn_client_number = -1
        self.unlearn_client_id = -1
        self.command = ""
        self.knowledge_eraser_rounds = 0

        self.mode = mode
        self.num_of_clients = num_of_clients
        self.num_server_rounds = num_server_rounds
        self.weight_factor_degradation_model = weight_factor_degradation_model
        self.weight_factor_global_model = weight_factor_global_model
        self.dataset_num_channels = dataset_num_channels
        self.dataset_num_classes = dataset_num_classes
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.plots_folder_path = plots_folder_path
        self.models_folder_path = models_folder_path

        self.degraded_model_parameters = None
        self.global_model_parameters = None
        self.client_plot = {}

    def configure_fit(self, server_round, parameters, client_manager):
        # Waiting till all clients are connected
        client_manager.wait_for(self.num_of_clients)

        config = {
            "current_round": server_round,
            "unlearn_client_number": self.unlearn_client_number,
            "command": self.command,
        }
        print("fit_ins.config", config)

        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        client_fit_pairs = []
        for client in clients:
            if (
                self.command == "global_model_restoration_and_degraded_model_unlearning"
                and self.unlearn_client_id == client.cid
            ):
                client_fit_pairs.append(
                    (client, FitIns(self.degraded_model_parameters, config))
                )
            elif self.command == "global_model_restoration":
                client_fit_pairs.append(
                    (client, FitIns(self.global_model_parameters, config))
                )
            else:
                client_fit_pairs.append((client, fit_ins))

        # Return client/config pairs
        return client_fit_pairs

    def configure_evaluate(self, server_round, parameters, client_manager):
        # Calling the parent class's configure_evaluate method
        client_evaluate_pairs = super().configure_evaluate(
            server_round, parameters, client_manager
        )

        for _, evaluate_ins in client_evaluate_pairs:
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

                initial_degraded_model_with_rand_parameters = get_weights(
                    get_net(
                        self.dataset_num_channels,
                        self.dataset_num_classes,
                        self.model_name,
                    )
                )
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
                or self.command == "global_model_restoration"
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

        for _, eval_res in results:
            client_number = eval_res.metrics["client_number"]

            if client_number not in self.client_plot:
                self.client_plot[client_number] = {}

            self.client_plot[client_number][server_round] = {
                "metrics": json.loads(str(eval_res.metrics).replace("'", '"')),
                "command": self.command,
            }

        if server_round == self.num_server_rounds:
            results_file_path = os.path.join(
                self.plots_folder_path,
                self.mode,
                f"{self.dataset_name}_{self.model_name}_results.json",
            )

            with open(results_file_path, "w", encoding="utf-8") as file:
                json.dump(self.client_plot, file)

            if self.mode == "federated_learning":
                model_file_path = os.path.join(
                    self.models_folder_path,
                    f"{self.dataset_name}_{self.model_name}_model.pkl",
                )
                with open(model_file_path, "wb") as file:
                    pickle.dump(self.global_model_parameters, file)

        if self.command == "degraded_model_initialization":
            self.command = "degraded_model_refinement"

        elif self.command == "global_model_unlearning":
            self.command = "global_model_restoration_and_degraded_model_unlearning"

        elif self.command == "global_model_restoration_and_degraded_model_unlearning":
            if self.knowledge_eraser_rounds == 2:
                self.command = "global_model_restoration"
            else:
                self.knowledge_eraser_rounds += 1
                self.command = "degraded_model_refinement"

        return super().aggregate_evaluate(server_round, results, failures)


def server_fn(context: Context):
    print("context.node_config", context)
    # Initialize model parameters
    mode = context.run_config.get("mode", None)
    dataset_num_channels = context.run_config.get("dataset-num-channels", None)
    dataset_num_classes = context.run_config.get("dataset-num-classes", None)
    model_name = context.run_config.get("model-name", None)
    dataset_name = context.run_config.get("dataset-name", None)
    ndarrays = get_weights(
        get_net(dataset_num_channels, dataset_num_classes, model_name)
    )

    plots_folder_path = context.run_config.get("plots-folder-path", None)
    models_folder_path = context.run_config.get("models-folder-path", None)

    parameters = None
    if mode == "federated_learning":
        parameters = ndarrays_to_parameters(ndarrays)
    elif mode == "federated_unlearning":
        model_file_path = os.path.join(
            models_folder_path, f"{dataset_name}_{model_name}_model.pkl"
        )
        with open(model_file_path, "rb") as file:
            parameters = pickle.load(file)

    fraction_evaluate = context.run_config.get("fraction-evaluate", None)
    num_of_clients = context.run_config.get("num-of-clients", None)
    num_server_rounds = context.run_config.get("num-server-rounds", None)
    weight_factor_degradation_model = context.run_config.get(
        "weight-factor-degradation-model", None
    )
    weight_factor_global_model = context.run_config.get(
        "weight-factor-global-model", None
    )

    # Define the strategy
    strategy = UnlearningFedAvg(
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
        mode=mode,
        num_of_clients=num_of_clients,
        num_server_rounds=num_server_rounds,
        weight_factor_degradation_model=weight_factor_degradation_model,
        weight_factor_global_model=weight_factor_global_model,
        dataset_num_channels=dataset_num_channels,
        dataset_num_classes=dataset_num_classes,
        model_name=model_name,
        dataset_name=dataset_name,
        plots_folder_path=plots_folder_path,
        models_folder_path=models_folder_path,
    )
    config = ServerConfig(num_rounds=context.run_config["num-server-rounds"])

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)

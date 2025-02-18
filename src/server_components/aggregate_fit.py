from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy.aggregate import aggregate_inplace

from src.server_components.helper import custom_aggregate


def aggregate_fit_retraining(results):
    filtered_results = []
    for client_proxy, fit_res in results:
        print("fit_res.metrics", fit_res.metrics)
        if fit_res.metrics.get("accuracy", 0) == 0:
            continue
        filtered_results.append((client_proxy, fit_res))

    aggregated_ndarrays = aggregate_inplace(filtered_results)

    parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

    metrics_aggregated = {}

    return parameters_aggregated, metrics_aggregated


def aggregate_fit_federated_unlearning(custom_fed_avg_instance, results):
    filtered_results = []
    for client_proxy, fit_res in results:
        print("fit_res.metrics", fit_res.metrics)
        if fit_res.metrics.get("unlearn_client_number", -1) != -1:
            custom_fed_avg_instance.unlearn_client_id = client_proxy.cid
            continue

        elif fit_res.metrics.get("accuracy", 0) == 0:
            continue

        elif (
            custom_fed_avg_instance.command
            == "global_model_restoration_and_degraded_model_unlearning"
        ) and custom_fed_avg_instance.unlearn_client_id == client_proxy.cid:
            custom_fed_avg_instance.degraded_model_parameters = fit_res.parameters
            continue

        filtered_results.append((client_proxy, fit_res))

    parameters_aggregated = ndarrays_to_parameters(aggregate_inplace(filtered_results))

    if custom_fed_avg_instance.command == "degraded_model_refinement":
        custom_fed_avg_instance.command = "global_model_unlearning"

    # Aggregate custom metrics if aggregation fn was provided
    metrics_aggregated = {}

    if custom_fed_avg_instance.command == "global_model_unlearning":
        custom_fed_avg_instance.degraded_model_parameters = parameters_aggregated
        custom_fed_avg_instance.global_model_parameters = ndarrays_to_parameters(
            custom_aggregate(
                [
                    (
                        parameters_to_ndarrays(
                            custom_fed_avg_instance.global_model_parameters
                        ),
                        custom_fed_avg_instance.weight_factor_global_model,
                    ),
                    (
                        parameters_to_ndarrays(parameters_aggregated),
                        1 - custom_fed_avg_instance.weight_factor_global_model,
                    ),
                ]
            )
        )
        return custom_fed_avg_instance.global_model_parameters, metrics_aggregated

    custom_fed_avg_instance.global_model_parameters = parameters_aggregated

    if (
        custom_fed_avg_instance.command
        == "global_model_restoration_and_degraded_model_unlearning"
    ):
        return custom_fed_avg_instance.degraded_model_parameters, metrics_aggregated

    return parameters_aggregated, metrics_aggregated

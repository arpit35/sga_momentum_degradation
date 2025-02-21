import os
import pickle
from functools import reduce

import numpy as np
from flwr.common import NDArrays


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


def save_model_to_disk(custom_fed_avg_instance):
    model_file_path = os.path.join(
        custom_fed_avg_instance.models_folder_path,
        f"{custom_fed_avg_instance.dataset_name}_{custom_fed_avg_instance.model_name}_model.pkl",
    )
    with open(model_file_path, "wb") as file:
        pickle.dump(custom_fed_avg_instance.global_model_parameters, file)

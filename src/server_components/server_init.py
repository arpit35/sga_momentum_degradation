import os
import pickle
from src.server_components.helper import custom_aggregate
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from src.ml_models.net import get_net
from src.ml_models.utils import get_weights

def server_init_federated_unlearning(models_folder_path, dataset_name, model_name, dataset_num_channels, dataset_num_classes, weight_factor_degradation_model):
    command = "degraded_model_refinement"
        model_file_path = os.path.join(
            models_folder_path, f"{dataset_name}_{model_name}_model.pkl"
        )
        with open(model_file_path, "rb") as file:
            global_model_parameters = pickle.load(file)

        initial_degraded_model_with_rand_parameters = get_weights(
            get_net(
                dataset_num_channels,
                dataset_num_classes,
                model_name,
            )
        )
        degraded_model_parameters = ndarrays_to_parameters(
            custom_aggregate(
                [
                    (
                        initial_degraded_model_with_rand_parameters,
                        weight_factor_degradation_model,
                    ),
                    (
                        parameters_to_ndarrays(global_model_parameters),
                        1 - weight_factor_degradation_model,
                    ),
                ]
            )
        )
        parameters = degraded_model_parameters

        return command, parameters, global_model_parameters, degraded_model_parameters

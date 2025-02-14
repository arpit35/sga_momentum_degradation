import os

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from src.data_loader import DataLoader
from src.ml_models.net import get_net, test, train
from src.ml_models.utils import get_weights, set_weights
from src.utils.logger import get_logger


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self,
        client_number,
        num_batches_each_round,
        batch_size,
        local_epochs,
        learning_rate,
        degraded_model_refinement_learning_rate,
        degraded_model_unlearning_rate,
        momentum,
        unlearning_trigger_client,
        unlearning_trigger_round,
        dataset_folder_path,
        dataset_num_channels,
        dataset_num_classes,
        dataset_input_feature,
        dataset_target_feature,
        model_name,
        model_input_image_size,
    ):
        super().__init__()
        self.net = get_net(dataset_num_channels, dataset_num_classes, model_name)
        self.client_number = client_number
        self.num_batches_each_round = num_batches_each_round
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.degraded_model_refinement_learning_rate = (
            degraded_model_refinement_learning_rate
        )
        self.degraded_model_unlearning_rate = degraded_model_unlearning_rate
        self.momentum = momentum
        self.unlearning_trigger_client = unlearning_trigger_client
        self.unlearning_trigger_round = unlearning_trigger_round
        self.client_folder_path = os.path.join(
            dataset_folder_path, f"client_{client_number}"
        )
        self.dataset_num_channels = dataset_num_channels
        self.dataset_input_feature = dataset_input_feature
        self.dataset_target_feature = dataset_target_feature
        self.model_input_image_size = model_input_image_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Configure logging
        self.logger = get_logger(f"{__name__}_Client_{client_number}", client_number)

        self.logger.info("Client %s initiated", self.client_number)

    def fit(self, parameters, config):
        sga = False

        # Fetching configuration settings from the server for the fit operation (server.configure_fit)
        current_round = config.get("current_round", 0)
        command = config.get("command", "")
        unlearn_client_number = config.get("unlearn_client_number", -1)

        self.logger.info("config: %s", config)
        self.logger.info("Client %s | Round %s", self.client_number, current_round)

        # target client is not taking part when the command is "degraded_model_refinement" or "global_model_restoration"
        if self.client_number == unlearn_client_number and (
            command == "degraded_model_refinement"
            or command == "global_model_restoration"
        ):
            self.logger.info(
                "Client %s is not taking part due to '%s' command",
                self.client_number,
                command,
            )
            return [], 0, {}

        results = {}

        # Unlearning initiated by the client
        if (
            current_round == self.unlearning_trigger_round
            and self.client_number == self.unlearning_trigger_client
        ):
            self.logger.info(
                "Unlearning initiated by the client: %s", self.client_number
            )
            results = {"unlearn_client_number": self.client_number}
            unlearn_client_number = self.client_number

        dataloader = DataLoader(
            dataset_input_feature=self.dataset_input_feature,
            dataset_num_channels=self.dataset_num_channels,
            model_input_image_size=self.model_input_image_size,
        )

        train_dataloader = dataloader.load_dataset_from_disk(
            "train_data",
            self.client_folder_path,
            self.num_batches_each_round,
            self.batch_size,
        )
        val_dataloader = dataloader.load_dataset_from_disk(
            "val_data",
            self.client_folder_path,
            self.num_batches_each_round,
            self.batch_size,
        )

        if command == "degraded_model_refinement":
            learning_rate = self.degraded_model_refinement_learning_rate
            momentum = self.momentum
        elif unlearn_client_number == self.client_number:
            sga = True
            learning_rate = self.degraded_model_unlearning_rate
            momentum = 0.0
        else:
            learning_rate = self.lr
            momentum = self.momentum

        set_weights(self.net, parameters)

        train_results = train(
            self.net,
            train_dataloader,
            val_dataloader,
            self.local_epochs,
            learning_rate,
            self.device,
            momentum,
            self.dataset_input_feature,
            self.dataset_target_feature,
            sga,
        )

        results.update(train_results)

        self.logger.info("sga: %s", sga)
        self.logger.info("results %s", results)
        self.logger.info("dataset_length %s", len(train_dataloader.dataset))
        self.logger.info("learning_rate: %s", learning_rate)
        self.logger.info("momentum: %s", momentum)

        return (
            get_weights(self.net),
            len(train_dataloader.dataset),
            results,
        )

    def _evaluate_model(self, parameters, file_name):
        set_weights(self.net, parameters)

        dataloader = DataLoader(
            dataset_input_feature=self.dataset_input_feature,
            dataset_num_channels=self.dataset_num_channels,
            model_input_image_size=self.model_input_image_size,
        )

        val_dataloader = dataloader.load_dataset_from_disk(
            file_name,
            self.client_folder_path,
            self.num_batches_each_round,
            self.batch_size,
        )
        loss, accuracy = test(
            self.net,
            val_dataloader,
            self.device,
            self.dataset_input_feature,
            self.dataset_target_feature,
        )
        val_dataset_length = len(val_dataloader.dataset)

        self.logger.info("loss: %s", loss)
        self.logger.info("accuracy: %s", accuracy)
        self.logger.info("val_dataset_length: %s", val_dataset_length)

        return loss, accuracy, val_dataset_length

    def evaluate(self, parameters, config):
        self.logger.info("config: %s", config)

        unlearn_client_number = config.get("unlearn_client_number", -1)
        command = config.get("command", "")

        if (
            command == "degraded_model_initialization"
            or command == "global_model_unlearning"
            or command == "global_model_restoration_and_degraded_model_unlearning"
            or command == "global_model_restoration"
        ) and self.client_number == unlearn_client_number:
            self.logger.info(
                "Client %s is not taking part in evaluation due to '%s' command",
                self.client_number,
                command,
            )
            loss, accuracy, val_dataset_length = self._evaluate_model(
                parameters, "poisoned_data"
            )

            return (
                0.0,
                0,
                {
                    "poisoned_data_loss": loss,
                    "poisoned_data_accuracy": accuracy,
                    "accuracy": 0,
                    "client_number": self.client_number,
                },
            )

        loss, accuracy, val_dataset_length = self._evaluate_model(
            parameters, "val_data"
        )

        return (
            loss,
            val_dataset_length,
            {
                "accuracy": accuracy,
                "client_number": self.client_number,
            },
        )


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    partition_id = context.node_config["partition-id"]
    num_batches_each_round = context.run_config["num-batches-each-round"]
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    degraded_model_refinement_learning_rate = context.run_config[
        "degraded-model-refinement-learning-rate"
    ]
    degraded_model_unlearning_rate = context.run_config[
        "degraded-model-unlearning-rate"
    ]
    momentum = context.run_config["momentum"]
    unlearning_trigger_client = context.run_config["unlearning-trigger-client"]
    unlearning_trigger_round = context.run_config["unlearning-trigger-round"]
    dataset_folder_path = context.run_config["dataset-folder-path"]
    dataset_num_channels = context.run_config["dataset-num-channels"]
    dataset_num_classes = context.run_config["dataset-num-classes"]
    dataset_input_feature = context.run_config["dataset-input-feature"]
    dataset_target_feature = context.run_config["dataset-target-feature"]
    model_name = context.run_config["model-name"]
    model_input_image_size = context.run_config["model-input-image-size"]

    # Return Client instance
    return FlowerClient(
        partition_id,
        num_batches_each_round,
        batch_size,
        local_epochs,
        learning_rate,
        degraded_model_refinement_learning_rate,
        degraded_model_unlearning_rate,
        momentum,
        unlearning_trigger_client,
        unlearning_trigger_round,
        dataset_folder_path,
        dataset_num_channels,
        dataset_num_classes,
        dataset_input_feature,
        dataset_target_feature,
        model_name,
        model_input_image_size,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)

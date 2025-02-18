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
        gradient_accumulation_steps,
        local_epochs,
        learning_rate,
        degraded_model_refinement_learning_rate,
        degraded_model_unlearning_rate,
        momentum,
        unlearning_trigger_client,
        dataset_folder_path,
        dataset_num_channels,
        dataset_num_classes,
        dataset_input_feature,
        dataset_target_feature,
        model_name,
        model_input_image_size,
        mode,
    ):
        super().__init__()
        self.net = get_net(dataset_num_channels, dataset_num_classes, model_name)
        self.client_number = client_number
        self.num_batches_each_round = num_batches_each_round
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.degraded_model_refinement_learning_rate = (
            degraded_model_refinement_learning_rate
        )
        self.degraded_model_unlearning_rate = degraded_model_unlearning_rate
        self.momentum = momentum
        self.unlearning_trigger_client = unlearning_trigger_client
        self.client_folder_path = os.path.join(
            dataset_folder_path, f"client_{client_number}"
        )
        self.dataset_num_channels = dataset_num_channels
        self.dataset_input_feature = dataset_input_feature
        self.dataset_target_feature = dataset_target_feature
        self.model_input_image_size = model_input_image_size
        self.mode = mode

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Configure logging
        self.logger = get_logger(f"{__name__}_Client_{client_number}", client_number)

        self.logger.info("Client %s initiated", self.client_number)

    def _get_training_config(self, unlearn_client_number, command):
        sga = False
        if command == "degraded_model_refinement":
            learning_rate = self.degraded_model_refinement_learning_rate
            momentum = self.momentum
        elif (
            unlearn_client_number == self.client_number
            and command == "global_model_restoration_and_degraded_model_unlearning"
        ):
            sga = True
            learning_rate = self.degraded_model_unlearning_rate
            momentum = 0.0
        else:
            learning_rate = self.lr
            momentum = self.momentum

        return learning_rate, momentum, sga

    def fit(self, parameters, config):
        # Fetching configuration settings from the server for the fit operation (server.configure_fit)
        current_round = config.get("current_round", 0)
        command = config.get("command", "")
        unlearn_client_number = config.get("unlearn_client_number", -1)

        self.logger.info("config: %s", config)
        self.logger.info("Client %s | Round %s", self.client_number, current_round)

        # target client is not taking part when the command is "degraded_model_refinement" or "global_model_restoration"
        if self.client_number == unlearn_client_number:
            if (
                command == "degraded_model_refinement"
                or command == "global_model_restoration"
                or self.mode == "retraining"
            ):
                self.logger.info(
                    "Client %s is not taking part due to '%s' command",
                    self.client_number,
                    command,
                )
                return [], 0, {}

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
            self.gradient_accumulation_steps,
        )
        val_dataloader = dataloader.load_dataset_from_disk(
            "val_data",
            self.client_folder_path,
            self.num_batches_each_round,
            self.batch_size,
            self.gradient_accumulation_steps,
        )

        learning_rate, momentum, sga = self._get_training_config(
            unlearn_client_number, command
        )

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
            self.gradient_accumulation_steps,
            sga,
        )

        self.logger.info("sga: %s", sga)
        self.logger.info("train_results %s", train_results)
        self.logger.info("dataset_length %s", len(train_dataloader.dataset))
        self.logger.info("learning_rate: %s", learning_rate)
        self.logger.info("momentum: %s", momentum)

        return (
            get_weights(self.net),
            len(train_dataloader.dataset),
            train_results,
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
            self.gradient_accumulation_steps,
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

    def _evaluate_federated_learning(self, parameters):
        loss, accuracy, val_dataset_length = self._evaluate_model(
            parameters, "val_data"
        )

        loss_poisoned, accuracy_poisoned, _ = self._evaluate_model(
            parameters, "poisoned_data"
        )

        return (
            loss,
            val_dataset_length,
            {
                "accuracy": accuracy,
                "client_number": self.client_number,
                "poisoned_data_loss": loss_poisoned,
                "poisoned_data_accuracy": accuracy_poisoned,
            },
        )

    def _evaluate_retraining(self, parameters):
        loss_poisoned, accuracy_poisoned, _ = self._evaluate_model(
            parameters, "poisoned_data"
        )

        return (
            0.0,
            0,
            {
                "accuracy": 0,
                "client_number": self.client_number,
                "poisoned_data_loss": loss_poisoned,
                "poisoned_data_accuracy": accuracy_poisoned,
            },
        )

    def _evaluate_federated_unlearning(self, parameters, command):
        if (
            command == "global_model_unlearning"
            or command == "global_model_restoration"
            or command == "global_model_restoration_and_degraded_model_unlearning"
        ):
            self.logger.info(
                "Client %s is not taking part in evaluation due to '%s' command",
                self.client_number,
                command,
            )
            loss, accuracy, _ = self._evaluate_model(parameters, "poisoned_data")

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

    def evaluate(self, parameters, config):
        self.logger.info("config: %s", config)

        unlearn_client_number = config.get("unlearn_client_number", -1)
        command = config.get("command", "")

        if self.client_number == unlearn_client_number:
            if self.mode == "federated_learning":
                return self._evaluate_federated_learning(parameters)
            elif self.mode == "retraining":
                return self._evaluate_retraining(parameters)
            elif self.mode == "federated_unlearning":
                return self._evaluate_federated_unlearning(parameters, command)

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

    partition_id = context.node_config.get("partition-id", None)
    num_batches_each_round = context.run_config.get("num-batches-each-round", None)
    batch_size = context.run_config.get("batch-size", None)
    gradient_accumulation_steps = context.run_config.get(
        "gradient-accumulation-steps", None
    )
    local_epochs = context.run_config.get("local-epochs", None)
    learning_rate = context.run_config.get("learning-rate", None)
    degraded_model_refinement_learning_rate = context.run_config.get(
        "degraded-model-refinement-learning-rate", None
    )
    degraded_model_unlearning_rate = context.run_config.get(
        "degraded-model-unlearning-rate", None
    )
    momentum = context.run_config.get("momentum", None)
    unlearning_trigger_client = context.run_config.get(
        "unlearning-trigger-client", None
    )
    dataset_folder_path = context.run_config.get("dataset-folder-path", None)
    dataset_num_channels = context.run_config.get("dataset-num-channels", None)
    dataset_num_classes = context.run_config.get("dataset-num-classes", None)
    dataset_input_feature = context.run_config.get("dataset-input-feature", None)
    dataset_target_feature = context.run_config.get("dataset-target-feature", None)
    model_name = context.run_config.get("model-name", None)
    model_input_image_size = context.run_config.get("model-input-image-size", None)
    mode = context.run_config.get("mode", None)

    # Return Client instance
    return FlowerClient(
        partition_id,
        num_batches_each_round,
        batch_size,
        gradient_accumulation_steps,
        local_epochs,
        learning_rate,
        degraded_model_refinement_learning_rate,
        degraded_model_unlearning_rate,
        momentum,
        unlearning_trigger_client,
        dataset_folder_path,
        dataset_num_channels,
        dataset_num_classes,
        dataset_input_feature,
        dataset_target_feature,
        model_name,
        model_input_image_size,
        mode,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)

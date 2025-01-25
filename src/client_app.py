import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from src.data_loader import dataset_length, load_client_data
from src.ml_models.net import Net, test, train
from src.ml_models.utils import get_weights, set_weights
from src.utils.logger import get_logger


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self,
        client_number,
        local_epochs,
        learning_rate,
        momentum,
    ):
        super().__init__()
        self.net = Net()
        self.client_number = client_number
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.momentum = momentum
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Configure logging
        self.logger = get_logger(f"{__name__}_Client_{client_number}", client_number)

        self.logger.info("Client %s initiated", self.client_number)

    def fit(self, parameters, config):
        sga = False

        set_weights(self.net, parameters)

        current_round = config.get("current_round", 0)
        command = config.get("command", "")
        unlearn_client_number = config.get("unlearn_client_number", -1)

        self.logger.info("config: %s", config)
        self.logger.info("Client %s | Round %s", self.client_number, current_round)

        if (
            command == "degraded_model_refinement"
            and self.client_number == unlearn_client_number
        ):
            self.logger.info(
                "Client %s is not taking part due to 'degraded_model_refinement' command",
                self.client_number,
            )
            return [], 0, {}

        results = {}

        if current_round == 101 and self.client_number == 0:
            self.logger.info(
                "Unlearning initiated by the client: %s", self.client_number
            )
            results = {"unlearn_client_number": self.client_number}
            sga = True

        if unlearn_client_number == self.client_number:
            sga = True

        self.logger.info("sga: %s", sga)

        train_batches = load_client_data("train", self.client_number, current_round)
        val_batches = load_client_data("val", self.client_number)

        train_results = train(
            self.net,
            train_batches,
            val_batches,
            self.local_epochs,
            self.lr,
            self.device,
            self.momentum,
            sga,
        )

        results.update(train_results)

        self.logger.info("results %s", results)
        self.logger.info("dataset_length %s", dataset_length(train_batches))

        return (
            get_weights(self.net),
            dataset_length(train_batches),
            results,
        )

    def evaluate(self, parameters, config):
        self.logger.info("config: %s", config)
        set_weights(self.net, parameters)

        val_batches = load_client_data("val", self.client_number)

        loss, accuracy = test(self.net, val_batches, self.device)
        return (
            loss,
            dataset_length(val_batches),
            {"accuracy": accuracy},
        )


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    partition_id = context.node_config["partition-id"]
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    momentum = context.run_config["momentum"]

    # Return Client instance
    return FlowerClient(
        partition_id,
        local_epochs,
        learning_rate,
        momentum,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)

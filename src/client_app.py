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
        client_id,
        local_epochs,
        learning_rate,
    ):
        super().__init__()
        self.net = Net()
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Configure logging
        self.logger = get_logger(f"{__name__}_Client_{client_id}", client_id)

        self.logger.info("Client %s initiated", self.client_id)

    def fit(self, parameters, config):
        sga = False

        set_weights(self.net, parameters)

        current_round = config.get("current_round", 0)

        unlearning_initiated_by_client_id = config.get(
            "unlearning_initiated_by_client_id", -1
        )

        self.logger.info("config: %s", config)
        self.logger.info("Client %s | Round %s", self.client_id, current_round)

        results = {}

        if current_round == 2 and self.client_id == 0:
            self.logger.info("Unlearning initiated by the client: %s", self.client_id)
            results = {"unlearning_initiated_by_client_id": self.client_id}
            sga = True

        if unlearning_initiated_by_client_id == self.client_id:
            sga = True

        self.logger.info("sga: %s", sga)

        train_batches = load_client_data("train", self.client_id, current_round)
        val_batches = load_client_data("val", self.client_id)

        train_results = train(
            self.net,
            train_batches,
            val_batches,
            self.local_epochs,
            self.lr,
            self.device,
            sga,
        )

        results.update(train_results)

        return (
            get_weights(self.net),
            dataset_length(train_batches),
            results,
        )

    def evaluate(self, parameters, config):
        self.logger.info("config: %s", config)
        set_weights(self.net, parameters)

        val_batches = load_client_data("val", self.client_id)

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

    # Return Client instance
    return FlowerClient(
        partition_id,
        local_epochs,
        learning_rate,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)

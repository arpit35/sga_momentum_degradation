import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from src.data_loader import load_client_data
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
        self.sga = False
        self.train_batches = None
        self.val_batches = None

        # Configure logging
        self.logger = get_logger(f"{__name__}_Client_{client_id}", client_id)

        self.logger.info("Client %s initiated", self.client_id)

    def dataset_length(self, dataset):
        return len(dataset[0]["image"]) * (len(dataset) - 1) + len(dataset[-1]["image"])

    def fit(self, parameters, config):
        set_weights(self.net, parameters)

        current_round = config.get("current_round", 0)

        unlearning_initiated_by_client_id = config.get(
            "unlearning_initiated_by_client_id", -1
        )

        self.logger.info("config: %s", config)
        self.logger.info("Client %s | Round %s", self.client_id, current_round)
        self.logger.info(
            "len(self.trainloader.dataset): %s", len(self.trainloader.dataset)
        )

        results["client_id"] = self.client_id

        if current_round == 2 and self.client_id == 0:
            self.logger.info("Unlearning initiated by the client: %s", self.client_id)
            results = {"unlearning_initiated_by_client_id": self.client_id}
            self.sga = True

        if unlearning_initiated_by_client_id == self.client_id:
            self.sga = True

        self.logger.info("sga: %s", self.sga)

        self.train_batches, self.val_batches = load_client_data(
            self.client_id, current_round
        )

        train_results = train(
            self.net,
            self.train_batches,
            self.val_batches,
            self.local_epochs,
            self.lr,
            self.device,
            self.sga,
        )

        results.update(train_results)

        return (
            get_weights(self.net),
            self.dataset_length(self.train_batches),
            results,
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.val_batches, self.device)
        return (
            loss,
            self.dataset_length(self.val_batches),
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

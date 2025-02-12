from src.data_loader import DataLoader
from src.scripts.helper import clear_folder_contents


def prepare_dataset(
    config: dict[str, str | int | bool], dataset_folder_path: str
) -> None:
    should_prepare_dataset = config.get("prepare-dataset", None)

    if not should_prepare_dataset:
        return

    print("Preparing dataset...")

    num_of_clients = int(config.get("num-of-clients", 0))
    num_server_rounds = int(config.get("num-server-rounds", 0))
    num_batches_each_round = int(config.get("num-batches-each-round", 0))
    batch_size = int(config.get("batch-size", 0))
    dataset_name = int(config.get("dataset-name", 0))
    alpha = float(config.get("data-loader-alpha", 0))
    unlearning_trigger_client = int(config.get("unlearning-trigger-client", 0))

    clear_folder_contents(dataset_folder_path)

    dataloader = DataLoader(dataset_name=str(dataset_name))

    dataloader.save_datasets(
        num_clients=num_of_clients,
        num_rounds=num_server_rounds,
        num_batches_each_round=num_batches_each_round,
        batch_size=batch_size,
        alpha=alpha,
        dataset_folder_path=dataset_folder_path,
        unlearning_trigger_client=unlearning_trigger_client,
    )

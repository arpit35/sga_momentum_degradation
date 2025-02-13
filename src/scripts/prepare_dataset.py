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
    dataset_name = str(config.get("dataset-name", None))
    alpha = float(config.get("data-loader-alpha", 0))
    unlearning_trigger_client = int(config.get("unlearning-trigger-client", 0))

    clear_folder_contents(dataset_folder_path)

    dataloader = DataLoader(dataset_name=dataset_name)

    dataloader.save_datasets(
        num_clients=num_of_clients,
        alpha=alpha,
        dataset_folder_path=dataset_folder_path,
        unlearning_trigger_client=unlearning_trigger_client,
    )

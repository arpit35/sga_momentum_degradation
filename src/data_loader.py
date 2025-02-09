import os
from collections import defaultdict

import numpy as np
import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from PIL import ImageDraw
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision import transforms


class DataLoader:
    def __init__(
        self,
        dataset_name: str,
    ):
        self.dataset_name = dataset_name
        if self.dataset_name == "mnist":
            self.pytorch_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )

    def _apply_transforms(self, batch):
        batch["image"] = [self.pytorch_transforms(img) for img in batch["image"]]
        return batch

    def _load_partition(self, num_clients: int, alpha: float):
        if self.dataset_name == "mnist":
            partition_by = "label"
        else:
            raise ValueError("Unknown dataset")

        partitioner = DirichletPartitioner(
            num_partitions=num_clients,
            partition_by=partition_by,
            alpha=alpha,
            self_balancing=True,
        )
        return FederatedDataset(
            dataset=self.dataset_name, partitioners={"train": partitioner}
        )

    def _add_trigger_to_sample(self, sample):
        """
        Applies a backdoor trigger to a single image sample (a PIL Image object).

        In this example, the trigger is a small 3x3 white square placed in the
        bottom-right corner. If the image is in grayscale ('L' mode), white is represented
        by the value 255; otherwise, for color images (e.g., 'RGB' mode) white is (255, 255, 255).

        Parameters:
            sample (PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: A copy of the input image with the backdoor trigger added.
        """
        # Create a copy of the image to avoid modifying the original
        sample_copy = sample.copy()

        # Define the trigger size (3x3 square)
        trigger_size = 3

        # Create a drawing context
        draw = ImageDraw.Draw(sample_copy)

        # Get the width and height of the image
        width, height = sample_copy.size  # Note: PIL returns (width, height)

        # Define the fill color based on the image mode
        if sample_copy.mode == "L":
            fill_color = 255  # white for grayscale images
        else:
            fill_color = (255, 255, 255)  # white for color images (RGB)

        # Define the coordinates for the rectangle (trigger)
        # Coordinates: (left, top, right, bottom)
        left = width - trigger_size
        top = height - trigger_size
        right = width
        bottom = height

        # Draw the rectangle (backdoor trigger)
        draw.rectangle([left, top, right, bottom], fill=fill_color)

        return sample_copy

    def _add_backdoor_to_partition(self, partition, batch_size, client_dir):
        print("partition", partition)

        # Get the labels from the partition
        labels = partition["label"]

        # Compute the unique classes and their counts
        unique_classes, counts = np.unique(labels, return_counts=True)

        print("unique_classes", unique_classes)
        print("counts", counts)

        # Identify the target class as the class with the most examples
        target_class = unique_classes[np.argmax(counts)]
        print("Target class selected for backdoor:", target_class)

        class_indices = defaultdict(list)
        for i, label in enumerate(labels):
            if label != target_class:
                class_indices[label].append(i)

        poison_indices = []  # list of indices that will be modified
        for cls, indices in class_indices.items():
            num_to_poison = int(0.7 * len(indices))

            # Randomly select indices without replacement
            selected = np.random.choice(indices, size=num_to_poison, replace=False)
            poison_indices.extend(selected)

        # Convert list to set for faster membership checks
        poison_indices = set(poison_indices)

        poisoned_samples = []

        def poison_sample(sample, idx):
            # The idx parameter is provided by the dataset's map function if with_indices=True.
            if idx in poison_indices:
                transformed_image = {}

                sample["image"] = self._add_trigger_to_sample(sample["image"])
                sample["label"] = torch.tensor(target_class, dtype=torch.long)

                transformed_image["image"] = self.pytorch_transforms(sample["image"])
                transformed_image["label"] = torch.tensor(
                    target_class, dtype=torch.long
                )
                poisoned_samples.append(transformed_image)
            return sample

        # Use the dataset's map function to modify the partition
        # The with_indices=True argument makes sure we get each example's index.
        partition = partition.map(poison_sample, with_indices=True)

        poisoned_data = TorchDataLoader(
            poisoned_samples, batch_size=batch_size, shuffle=True
        )

        poisoned_data_path = os.path.join(client_dir, "poisoned_data.pt")
        torch.save(list(poisoned_data), poisoned_data_path)

        return partition

    def save_datasets(
        self,
        num_clients: int,
        num_rounds: int,
        num_batches_each_round: int,
        batch_size: int,
        alpha: float,
        clients_dataset_folder_path: str,
        unlearning_trigger_client: int,
    ):
        fds = self._load_partition(num_clients, alpha)
        for client_id in range(num_clients):
            client_dir = os.path.join(
                clients_dataset_folder_path, f"client_{client_id}"
            )
            os.makedirs(client_dir, exist_ok=True)

            if unlearning_trigger_client == client_id:
                partition = self._add_backdoor_to_partition(
                    fds.load_partition(client_id), batch_size, client_dir
                )
            else:
                partition = fds.load_partition(client_id)

            partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
            partition_train_test = partition_train_test.with_transform(
                self._apply_transforms
            )

            valloader = TorchDataLoader(
                partition_train_test["test"], batch_size=batch_size
            )

            val_path = os.path.join(client_dir, "val_data.pt")
            torch.save(list(valloader), val_path)

            # Train batches for each round
            train_data = partition_train_test["train"]
            train_indices = np.arange(len(train_data))

            for round_num in range(num_rounds):
                round_batches = []

                for _ in range(num_batches_each_round):
                    selected_indices = np.random.choice(
                        train_indices, size=batch_size, replace=False
                    )
                    train_batch = {
                        key: (
                            torch.stack(
                                [train_data[int(idx)][key] for idx in selected_indices]
                            )
                            if isinstance(train_data[0][key], torch.Tensor)
                            else torch.tensor(
                                [train_data[int(idx)][key] for idx in selected_indices]
                            )
                        )
                        for key in train_data[0]
                    }
                    round_batches.append(train_batch)

                round_path = os.path.join(
                    client_dir, f"train_data_for_round_{round_num + 1}.pt"
                )
                torch.save(round_batches, round_path)


def load_client_data(type: str, client_id: int, current_round: int = None):
    client_dir = os.path.join("src", "clients_dataset", f"client_{client_id}")

    if type == "val":
        # Load validation dataset
        val_path = os.path.join(client_dir, "val_data.pt")
        return torch.load(val_path, weights_only=False)

    if type == "poisoned":
        # Load poisoned dataset
        poisoned_path = os.path.join(client_dir, "poisoned_data.pt")
        return torch.load(poisoned_path, weights_only=False)

    if type == "train":
        # Load train dataset for the specified round
        round_path = os.path.join(
            client_dir, f"train_data_for_round_{current_round}.pt"
        )
        return torch.load(round_path, weights_only=False)


def dataset_length(dataset):
    return len(dataset[0]["image"]) * (len(dataset) - 1) + len(dataset[-1]["image"])

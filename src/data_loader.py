import os
from collections import defaultdict

import numpy as np
from datasets import load_from_disk
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision import transforms


class DataLoader:
    def __init__(
        self,
        dataset_input_feature: str,
        dataset_target_feature: str = None,
        dataset_name: str = None,
        dataset_num_channels: int = None,
        model_input_image_size: int = None,
    ):
        self.dataset_name = dataset_name
        self.dataset_input_feature = dataset_input_feature
        self.dataset_target_feature = dataset_target_feature

        if dataset_name is None:
            self.pytorch_transforms = self._get_transform(
                dataset_num_channels, model_input_image_size
            )

    def _get_transform(self, dataset_num_channels, model_input_image_size):
        if dataset_num_channels == 1:
            # For grayscale images
            normalize = transforms.Normalize((0.5,), (0.5,))
        elif dataset_num_channels == 3:
            # For RGB images
            normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            raise ValueError(f"Unsupported number of channels: {dataset_num_channels}")

        transform = transforms.Compose(
            [
                transforms.Resize((model_input_image_size, model_input_image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )
        return transform

    def _apply_transforms(self, batch):
        batch[self.dataset_input_feature] = [
            self.pytorch_transforms(img) for img in batch[self.dataset_input_feature]
        ]
        return batch

    def _load_partition(self, num_clients: int, alpha: float):
        partitioner = DirichletPartitioner(
            num_partitions=num_clients,
            partition_by=self.dataset_target_feature,
            alpha=alpha,
            self_balancing=True,
        )
        return FederatedDataset(
            dataset=self.dataset_name, partitioners={"train": partitioner}
        )

    def _add_trigger_to_sample(self, sample):
        """
        Applies a backdoor trigger to a single image sample (a PIL Image object).

        In this example, the trigger is a small 3x3 pattern placed in the
        bottom-right corner of the image. The pattern is defined as follows:

        0    0    255
        0    255  0
        255  0    255

        For grayscale images ('L' mode), each pixel is set to the scalar value.
        For color images (e.g., 'RGB' mode), each pixel is set by repeating the
        scalar value across all three channels (i.e., (v, v, v)).

        Parameters:
            sample (PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: A copy of the input image with the backdoor trigger added.
        """
        # Create a copy of the image to avoid modifying the original
        sample_copy = sample.copy()

        # Define the trigger size (3x3)
        trigger_size = 3

        # Get the width and height of the image
        width, height = sample_copy.size  # PIL returns (width, height)

        # Define the custom 3x3 pattern
        pattern = [[0, 0, 255], [0, 255, 0], [255, 0, 255]]

        # Loop over the 3x3 region at the bottom-right of the image
        for i in range(trigger_size):  # vertical offset (rows)
            for j in range(trigger_size):  # horizontal offset (columns)
                # Compute the exact pixel location in the image
                x = width - trigger_size - 2 + j
                y = height - trigger_size - 2 + i

                # For grayscale images, assign the scalar value.
                # For RGB (or other color modes), replicate the value across channels.
                if sample_copy.mode == "L":
                    sample_copy.putpixel((x, y), pattern[i][j])
                else:
                    sample_copy.putpixel((x, y), (pattern[i][j],) * 3)

        return sample_copy

    def _add_backdoor_to_partition(self, partition, data_type, target_class):

        # Get the labels from the partition
        labels = partition[self.dataset_target_feature]

        class_indices = defaultdict(list)
        for i, label in enumerate(labels):
            if label != target_class:
                class_indices[label].append(i)

        poison_indices = []  # list of indices that will be modified
        for _, indices in class_indices.items():
            num_to_poison = int(0.75 * len(indices))

            # Randomly select indices without replacement
            selected = np.random.choice(indices, size=num_to_poison, replace=False)
            poison_indices.extend(selected)

        # Convert list to set for faster membership checks
        poison_indices = set(poison_indices)

        def _add_trigger_if_poisoned(sample, idx):
            if idx in poison_indices:
                sample[self.dataset_input_feature] = self._add_trigger_to_sample(
                    sample[self.dataset_input_feature]
                )
                sample[self.dataset_target_feature] = target_class
                sample["poisoned"] = True  # Add a new column to track poisoned samples
            else:
                sample["poisoned"] = False
            return sample

        # Apply transformation with a new "poisoned" column
        partition = partition.map(
            _add_trigger_if_poisoned, with_indices=True, load_from_cache_file=False
        )

        if data_type == "train":
            # Filter poisoned samples based on the new "poisoned" column
            poisoned_partition = partition.filter(
                lambda sample: sample["poisoned"], load_from_cache_file=False
            )

            return (
                partition.remove_columns(["poisoned"]),
                poisoned_partition.remove_columns(["poisoned"]),
            )

        return partition.remove_columns(["poisoned"])

    def save_datasets_to_disk(
        self,
        num_clients: int,
        alpha: float,
        dataset_folder_path: str,
        unlearning_trigger_client: int,
    ):
        fds = self._load_partition(num_clients, alpha)
        for client_id in range(num_clients):
            client_dataset_folder_path = os.path.join(
                dataset_folder_path, f"client_{client_id}"
            )
            os.makedirs(client_dataset_folder_path, exist_ok=True)

            partition = fds.load_partition(client_id)

            labels = partition[self.dataset_target_feature]
            # Compute the unique classes and their counts
            unique_classes, counts = np.unique(labels, return_counts=True)

            # Filter out classes with only one row
            classes_to_keep = set(unique_classes[counts > 1])
            partition = partition.filter(
                lambda example, keep=classes_to_keep: example[
                    self.dataset_target_feature
                ]
                in keep,
                load_from_cache_file=False,
            )

            partition_train_test = partition.train_test_split(
                test_size=0.2, seed=42, stratify_by_column=self.dataset_target_feature
            )

            if unlearning_trigger_client == client_id:
                # Identify the target class as the class with the most examples
                valid_classes = unique_classes[counts > 100]
                valid_counts = counts[counts > 100]
                target_class = valid_classes[np.argmin(valid_counts)]
                print("Target class selected for backdoor:", target_class)

                train_partition, poisoned_partition = self._add_backdoor_to_partition(
                    partition_train_test["train"], "train", target_class
                )
                test_partition = self._add_backdoor_to_partition(
                    partition_train_test["test"], "test", target_class
                )

                poisoned_path = os.path.join(
                    client_dataset_folder_path, "poisoned_data"
                )

                clean_train_path = os.path.join(
                    client_dataset_folder_path, "clean_train_data"
                )

                poisoned_partition.save_to_disk(poisoned_path)
                partition_train_test["train"].save_to_disk(clean_train_path)
            else:
                train_partition = partition_train_test["train"]
                test_partition = partition_train_test["test"]

            train_path = os.path.join(client_dataset_folder_path, "train_data")
            val_path = os.path.join(client_dataset_folder_path, "val_data")

            train_partition.save_to_disk(train_path)
            test_partition.save_to_disk(val_path)

    def load_dataset_from_disk(
        self,
        file_name: str,
        client_folder_path,
        num_batches_each_round,
        batch_size,
        gradient_accumulation_steps,
    ):
        client_file_path = os.path.join(client_folder_path, file_name)

        client_dataset = load_from_disk(client_file_path).with_transform(
            self._apply_transforms
        )
        client_dataset_lenght = len(client_dataset)
        # Calculate the total number of samples to select
        if file_name == "train_data":
            total_samples = num_batches_each_round * batch_size
        else:
            total_samples = int(
                num_batches_each_round * (batch_size / gradient_accumulation_steps)
            )

        if client_dataset_lenght > total_samples:
            # Randomly select indices
            all_indices = np.arange(len(client_dataset))
            selected_indices = np.random.choice(
                all_indices, total_samples, replace=False
            )
            client_dataset = client_dataset.select(selected_indices)

        # Create a PyTorch DataLoader
        data_loader = TorchDataLoader(
            client_dataset,
            batch_size=int(batch_size / gradient_accumulation_steps),
            shuffle=True,
        )

        return data_loader

import matplotlib.pyplot as plt
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision import transforms


class DataLoader:
    def __init__(
        self, dataset_name: str, num_clients: int, batch_size: int, alpha: float
    ):
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.alpha = alpha
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

    def load_partition(self, partition_id: int):
        if self.dataset_name == "mnist":
            partition_by = "label"
        else:
            raise ValueError("Unknown dataset")

        partitioner = DirichletPartitioner(
            num_partitions=self.num_clients,
            partition_by=partition_by,
            alpha=self.alpha,
            self_balancing=True,
        )
        fds = FederatedDataset(
            dataset=self.dataset_name, partitioners={"train": partitioner}
        )
        partition = fds.load_partition(partition_id)
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        partition_train_test = partition_train_test.with_transform(
            self._apply_transforms
        )

        trainloader = TorchDataLoader(
            partition_train_test["train"], batch_size=self.batch_size, shuffle=True
        )
        valloader = TorchDataLoader(
            partition_train_test["test"], batch_size=self.batch_size
        )
        return trainloader, valloader

    def load_test_set(self):
        # Specify partitioners as empty (since test set does not need partitioning)
        fds = FederatedDataset(dataset=self.dataset_name, partitioners={})
        testset = fds.load_split("test").with_transform(self._apply_transforms)
        testloader = TorchDataLoader(testset, batch_size=self.batch_size)
        return testloader

    def print_partition_sizes(self, trainloader, valloader, testloader):
        train_size = len(trainloader.dataset)
        val_size = len(valloader.dataset)
        test_size = len(testloader.dataset)

        print(f"Training set size: {train_size}")
        print(f"Validation set size: {val_size}")
        print(f"Test set size: {test_size}")

    def visualize_batch(self, dataloader):

        batch = next(iter(dataloader))
        images, labels = batch["image"], batch["label"]

        # Reshape and convert images to NumPy arrays
        images = images.permute(0, 2, 3, 1).numpy()
        images = images / 2 + 0.5  # Denormalize

        # Create a figure and a grid of subplots
        fig, axs = plt.subplots(4, 8, figsize=(12, 6))
        for i, ax in enumerate(axs.flat):
            if i < len(images):
                ax.imshow(images[i])
                ax.set_title(str(labels[i].item()))
                ax.axis("off")

        plt.tight_layout()
        plt.show()

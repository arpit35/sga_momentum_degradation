import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_loader import dataset_length


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm1d(120)
        self.bn4 = nn.BatchNorm1d(84)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        return self.fc3(x)


fds = None  # Cache FederatedDataset


def train(
    net, train_batches, val_batches, epochs, learning_rate, device, momentum, SGA=False
):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(
        net.parameters(), lr=learning_rate, momentum=0.9, maximize=SGA
    )
    net.train()

    for _ in range(epochs):
        for batch in train_batches:
            images = batch["image"]
            labels = batch["label"]
            optimizer.zero_grad()
            criterion(net(images.to(device)), labels.to(device)).backward()
            optimizer.step()

    val_loss, val_acc = test(net, val_batches, device)

    results = {
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


def test(net, test_batch, device):
    """Validate the model on the test set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in test_batch:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / dataset_length(test_batch)
    loss = loss / dataset_length(test_batch)
    return loss, accuracy

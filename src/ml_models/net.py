import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.pool = nn.MaxPool2d(2, 2, 1, 2)

        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 2)
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 2)
        self.conv4 = nn.Conv2d(32, 64, 3, 1, 2)

        self.fc1 = nn.Linear(64 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm1d(120)
        self.bn6 = nn.BatchNorm1d(84)

    def forward(self, x):
        # print("Input shape:", x.shape)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # print("After conv1 and pool:", x.shape)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # print("After conv2 and pool:", x.shape)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # print("After conv3 and pool:", x.shape)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # print("After conv4 and pool:", x.shape)
        x = x.view(-1, 64 * 4 * 4)
        # print("After view:", x.shape)
        x = F.relu(self.bn5(self.fc1(x)))
        # print("After fc1:", x.shape)
        x = F.relu(self.bn6(self.fc2(x)))
        # print("After fc2:", x.shape)
        x = self.fc3(x)
        # print("Output shape:", x.shape)
        return x


def get_net(dataset_num_channels, dataset_num_classes, model_name):
    model_func = getattr(models, model_name, None)
    if model_func is None:
        raise ValueError(f"Model '{model_name}' not found in torchvision.models")

    # Load model without pretrained weights
    model = model_func(weights=None)

    if re.search(r"resnet", model_name, re.IGNORECASE):
        # Modify the first convolution layer
        model.conv1 = nn.Conv2d(
            in_channels=dataset_num_channels,  # Change from 3 to 1
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Modify the final fully connected layer
        model.fc = nn.Linear(model.fc.in_features, dataset_num_classes)

    return model


def train(
    net,
    train_batches,
    val_batches,
    epochs,
    learning_rate,
    device,
    momentum,
    dataset_input_feature,
    dataset_target_feature,
    gradient_accumulation_steps,
    sga=False,
):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(
        net.parameters(), lr=learning_rate, momentum=momentum, maximize=sga
    )
    net.train()

    for _ in range(epochs):
        optimizer.zero_grad()
        for i, batch in enumerate(train_batches):
            images = batch[dataset_input_feature].to(device)
            labels = batch[dataset_target_feature].to(device)
            loss = criterion(net(images), labels)

            # Normalize loss to account for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()

            # Update weights after accumulating gradients over 'accumulation_steps' mini-batches
            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            optimizer.step()

        # Handle the case where the number of batches isn't divisible by accumulation_steps
        if len(train_batches) % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

    val_loss, val_acc = test(
        net,
        val_batches,
        device,
        dataset_input_feature,
        dataset_target_feature,
    )

    results = {
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


def test(
    net,
    test_batches,
    device,
    dataset_input_feature,
    dataset_target_feature,
):
    """Validate the model on the test set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in test_batches:
            images = batch[dataset_input_feature].to(device)
            labels = batch[dataset_target_feature].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(test_batches.dataset)
    loss = loss / len(test_batches.dataset)
    return loss, accuracy

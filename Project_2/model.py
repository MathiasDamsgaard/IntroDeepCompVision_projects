import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights


class BaselineClassifier(nn.Module):
    def __init__(self, n_classes: int = 10) -> None:
        """Initialize the VideoClassifier with a pretrained ResNet18 backbone.

        Args:
            n_classes (int): Number of output classes for classification. Default is 10.

        The model uses transfer learning with a frozen ResNet18 backbone and only
        trains the final fully connected layer for the specific classification task.

        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transfer learning with pretrained ResNet18
        self.base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace the final fully connected layer for our 10 classes
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, n_classes)
        self.base_model.to(self.device)

        # Freeze backbone (all layers except the final fc layer)
        # Only train the final classification layer
        for name, param in self.base_model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

        # Optimizer on trainable parameters only (just the fc layer)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.base_model.parameters()), lr=1e-3, weight_decay=1e-2
        )

        # CrossEntropyLoss expects raw logits (not softmax outputs)
        self.criterion = nn.CrossEntropyLoss()

    def fit(
        self, num_epochs: int, train_loader: DataLoader, test_loader: DataLoader
    ) -> tuple[list[float], list[float]]:
        """Train the model on the training set and evaluate on validation set.

        Args:
            num_epochs (int): Number of training epochs to run.
            train_loader (DataLoader): DataLoader for the training dataset.
            test_loader (DataLoader): DataLoader for the validation dataset.

        Returns:
            tuple: (train_acc, test_acc) - Lists of training and validation accuracies
                   for each epoch, with values in range [0, 1].

        """
        train_acc, test_acc = [], []
        self.base_model.to(self.device)

        for _i in range(num_epochs):
            # For each epoch
            self.base_model.train()
            train_correct = 0
            total_train_samples = 0

            for data, target in train_loader:
                data_gpu, target_gpu = data.to(self.device), target.to(self.device)

                # Zero the gradients computed for each weight
                self.optimizer.zero_grad()

                # Forward pass (use GPU tensors)
                output = self.base_model(data_gpu)

                # Compute the loss (use GPU targets)
                loss = self.criterion(output, target_gpu)

                # Backward pass through the network
                loss.backward()

                # Update the weights
                self.optimizer.step()

                # Compute how many were correctly classified
                predicted = output.argmax(1)
                train_correct += (predicted == target_gpu).sum().cpu().item()

                # Track how many samples we've seen
                total_train_samples += target_gpu.size(0)

            # Compute the validation accuracy
            self.base_model.eval()
            test_correct = 0
            total_test_samples = 0

            for data, target in test_loader:
                data_gpu, target_gpu = data.to(self.device), target.to(self.device)
                with torch.no_grad():
                    output = self.base_model(data_gpu)
                predicted = output.argmax(1)
                test_correct += (predicted == target_gpu).sum().cpu().item()
                total_test_samples += target_gpu.size(0)

            # Use counted totals instead of len(dataset)
            train_acc.append(train_correct / total_train_samples)
            test_acc.append(test_correct / total_test_samples)

        return train_acc, test_acc

    def evaluate(self, data: torch.Tensor) -> torch.Tensor:
        """Evaluate the model on given data and return raw logits.

        Args:
            data (torch.Tensor): Input batch of images with shape [batch_size, C, H, W].

        Returns:
            torch.Tensor: Raw logits (unnormalized predictions) with shape [batch_size, n_classes].
                         Apply softmax to convert to probabilities, or argmax to get class predictions.

        """
        self.base_model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            output = self.base_model(data)  # Returns logits (no softmax)
        return output.cpu()

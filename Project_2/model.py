import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.video import R3D_18_Weights, r3d_18


class BaselineClassifier(nn.Module):
    def __init__(self, n_classes: int = 10, is_flow: bool = False) -> None:
        """Initialize the VideoClassifier with a pretrained ResNet18 backbone.

        Args:
            n_classes (int): Number of output classes for classification. Default is 10.
            is_flow (bool): Whether the model is used in optical flow task. Default is False.

        The model uses transfer learning with a frozen ResNet18 backbone and only
        trains the final fully connected layer for the specific classification task.

        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transfer learning with pretrained ResNet18
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)

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

        # Loss function
        self.criterion = nn.NLLLoss() if is_flow else nn.CrossEntropyLoss()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the baseline classifier.

        Args:
            x (torch.Tensor): Input tensor of shape [T, C, H, W]

        Returns:
            torch.Tensor: Output tensor of shape [T, n_classes]

        """
        return self.base_model(x.to(self.device))

    def evaluate(self, data: torch.Tensor) -> torch.Tensor:
        """Evaluate the model on given data and return softmaxed predictions.

        Args:
            data (torch.Tensor): Input batch of images with shape [batch_size, C, H, W].

        Returns:
            torch.Tensor: Softmaxed predictions with shape [batch_size, n_classes].

        """
        self.base_model.eval()
        data = data.to(self.device)
        predictions = []

        for frames in data:
            with torch.no_grad():
                output = self.base_model(frames)
                # Convert logits to probabilities along the class dimension
                probs = F.softmax(output, dim=1)
                # Average across temporal or clip dimension if multiple predictions per sample
                mean_probs = probs.mean(dim=0)
            predictions.append(mean_probs.cpu())

        return torch.stack(predictions)


class EarlyFusionCNN(nn.Module):
    """Early fusion CNN model for video classification.

    This model stacks all frames along the channel dimension before feeding them to the CNN.
    For example, with 3-channel frames and 10 frames, the input tensor would have 30 channels.
    """

    def __init__(self, n_classes: int = 10, n_frames: int = 10, n_channels: int = 3, is_flow: bool = False) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Modified ResNet with expanded input channels to accommodate stacked frames
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Modify the first conv layer to handle n_frames * 3 channels instead of just 3
        self.base_model.conv1.in_channels = n_channels * n_frames
        self.base_model.conv1.weight = nn.Parameter(torch.cat([self.base_model.conv1.weight] * n_frames, dim=1))

        # Replace the final fully connected layer for our classes
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

        # Loss function
        self.criterion = nn.NLLLoss() if is_flow else nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for early fusion model.

        Args:
            x: Tensor of shape [batch_size, T, C, H, W]
               depending on how the frames are stacked

        Returns:
            Tensor of shape [batch_size, n_classes] containing class logits

        """
        # Make sure the input is on the same device as the model
        x = x.to(self.device)

        # [batch_size, T, C, H, W] format
        batch_size, t, c, h, w = x.shape

        # Reshape to [batch_size, T * C, H, W]
        x = x.reshape(batch_size, t * c, h, w)

        return self.base_model(x)

    def fit(
        self, num_epochs: int, train_loader: DataLoader, test_loader: DataLoader
    ) -> tuple[list[float], list[float]]:
        """Train the model on the training set and evaluate on validation set.

        Similar to VideoClassifier.fit but expects frames to be stacked along channel dimension.
        """
        train_acc, test_acc = [], []
        self.base_model.to(self.device)

        for _ in range(num_epochs):
            # Training phase
            self.base_model.train()
            train_correct = 0
            total_train_samples = 0

            for data, target in train_loader:
                data_gpu, target_gpu = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.forward(data_gpu)
                loss = self.criterion(output, target_gpu)
                loss.backward()
                self.optimizer.step()

                predicted = output.argmax(1)
                train_correct += (predicted == target_gpu).sum().cpu().item()
                total_train_samples += target_gpu.size(0)

            # Validation phase
            self.base_model.eval()
            test_correct = 0
            total_test_samples = 0

            with torch.no_grad():
                for data, target in test_loader:
                    data_gpu, target_gpu = data.to(self.device), target.to(self.device)
                    output = self.forward(data_gpu)
                    predicted = output.argmax(1)
                    test_correct += (predicted == target_gpu).sum().cpu().item()
                    total_test_samples += target_gpu.size(0)

            train_acc.append(train_correct / total_train_samples)
            test_acc.append(test_correct / total_test_samples)

        return train_acc, test_acc

    def evaluate(self, data: torch.Tensor) -> torch.Tensor:
        """Evaluate the model on given data and return softmaxed probabilities."""
        self.base_model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            output = self.forward(data)
            probs = F.softmax(output, dim=1)
        return probs.cpu()


class LateFusionCNN(nn.Module):
    """Late fusion CNN model for video classification.

    This model processes each frame independently through a CNN and then aggregates
    the features or predictions at a later stage.
    """

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Frame feature extractor using pretrained ResNet18
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Use a Linear layer that behaves like Identity (satisfies type checker)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])

        # Temporal fusion: After getting features from each frame
        self.temporal_fusion = nn.Sequential(
            # nn.Linear(10*512*2*2, 10_000), nn.ReLU(), nn.Dropout(0.5), nn.Linear(10_000, n_classes)
            nn.Linear(10 * 512 * 2 * 2, n_classes)
        )

        # Freeze backbone except last layer
        for _, param in self.base_model.named_parameters():
            param.requires_grad = False

        # Move to device
        self.base_model.to(self.device)
        self.temporal_fusion.to(self.device)

        # Optimizer and loss
        trainable_params = list(self.temporal_fusion.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, lr=1e-3, weight_decay=1e-2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass for late fusion model.

        Args:
            x: List of frame tensors, each of shape [batch_size, C, H, W]

        Returns:
            Tensor of shape [batch_size, n_classes] containing class logits

        """
        # Extract features from each frame
        outputs = []
        for frames in x:
            frames_gpu = frames.to(self.device)
            features = self.base_model(frames_gpu)
            features = torch.flatten(features)
            outputs.append(self.temporal_fusion(features))

        # Final classification
        return torch.stack(outputs)

    def fit(
        self, num_epochs: int, train_loader: DataLoader, test_loader: DataLoader
    ) -> tuple[list[float], list[float]]:
        """Train the model on the training set and evaluate on validation set."""
        train_acc, test_acc = [], []
        self.base_model.eval()

        for _ in range(num_epochs):
            # Training phase
            self.temporal_fusion.train()
            train_correct = 0
            total_train_samples = 0

            for batch, targets in train_loader:
                targets_gpu = targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.forward(batch)
                loss = self.criterion(outputs, targets_gpu)
                loss.backward()
                self.optimizer.step()

                predicted = outputs.argmax(1)
                train_correct += (predicted == targets_gpu).sum().cpu().item()
                total_train_samples += targets.size(0)

            # Validation phase
            self.temporal_fusion.eval()
            test_correct = 0
            total_test_samples = 0

            with torch.no_grad():
                for batch, targets in test_loader:
                    targets_gpu = targets.to(self.device)

                    outputs = self.forward(batch)

                    predicted = outputs.argmax(1)
                    test_correct += (predicted == targets_gpu).sum().cpu().item()
                    total_test_samples += targets.size(0)

            train_acc.append(train_correct / total_train_samples)
            test_acc.append(test_correct / total_test_samples)

        return train_acc, test_acc

    def evaluate(self, data: list[torch.Tensor]) -> torch.Tensor:
        """Evaluate the model on given data and return softmaxed probabilities for late fusion."""
        self.base_model.eval()
        self.temporal_fusion.eval()
        with torch.no_grad():
            output = self.forward(data)  # forward should return raw logits
            probs = F.softmax(output, dim=1)  # convert logits to probabilities
        return probs.cpu()


class CNN3D(nn.Module):
    """3D CNN model for video classification.

    This model uses 3D convolutions (ResNet-18 backbone) to jointly capture
    spatial and temporal information in videos.
    """

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained 3D ResNet-18
        self.base_model = r3d_18(weights=R3D_18_Weights.DEFAULT)

        # Replace final fully connected layer
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, n_classes)
        self.base_model.to(self.device)

        # Freeze backbone (all layers except final fc)
        for name, param in self.base_model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

        # Optimizer only on trainable params (fc layer)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.base_model.parameters()),
            lr=1e-3,
            weight_decay=1e-2,
        )

        # CrossEntropyLoss for classification
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for 3D CNN.

        Args:
            x (torch.Tensor): Input of shape [batch_size, C, T, H, W]

        Returns:
            torch.Tensor: Class logits [batch_size, n_classes]

        """
        x = x.to(self.device)
        return self.base_model(x)

    def fit(
        self, num_epochs: int, train_loader: DataLoader, test_loader: DataLoader
    ) -> tuple[list[float], list[float]]:
        """Train the model on the training set and evaluate on validation set."""
        train_acc, test_acc = [], []
        self.base_model.to(self.device)

        for _ in range(num_epochs):
            # Training phase
            self.base_model.train()
            train_correct = 0
            total_train_samples = 0

            for data, target in train_loader:
                data_gpu, target_gpu = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.base_model(data_gpu)
                loss = self.criterion(output, target_gpu)
                loss.backward()
                self.optimizer.step()

                predicted = output.argmax(1)
                train_correct += (predicted == target_gpu).sum().cpu().item()
                total_train_samples += target_gpu.size(0)

            # Validation phase
            self.base_model.eval()
            test_correct = 0
            total_test_samples = 0

            with torch.no_grad():
                for data, target in test_loader:
                    data_gpu, target_gpu = data.to(self.device), target.to(self.device)
                    output = self.base_model(data_gpu)
                    predicted = output.argmax(1)
                    test_correct += (predicted == target_gpu).sum().cpu().item()
                    total_test_samples += target_gpu.size(0)

            train_acc.append(train_correct / total_train_samples)
            test_acc.append(test_correct / total_test_samples)

        return train_acc, test_acc

    def evaluate(self, data: torch.Tensor) -> torch.Tensor:
        """Evaluate the model on given data and return softmaxed probabilities."""
        self.base_model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            output = self.base_model(data)  # raw logits
            probs = F.softmax(output, dim=1)  # convert to probabilities
        return probs.cpu()


class FlowCNN(nn.Module):
    def __init__(
        self, n_classes: int = 10, n_frames: int = 10, n_channels: int = 2, lr: float = 0.001, weight_decay: float = 0
    ) -> None:
        super().__init__()
        self.frame_model = BaselineClassifier(n_classes=n_classes, is_flow=True)
        self.flow_model = EarlyFusionCNN(
            n_classes=n_classes, n_frames=n_frames - 1, n_channels=n_channels, is_flow=True
        )
        self.frame_model.optimizer = torch.optim.Adam(self.frame_model.parameters(), lr=lr, weight_decay=weight_decay)
        self.flow_model.optimizer = torch.optim.Adam(self.flow_model.parameters(), lr=lr, weight_decay=weight_decay)

        # Loss function
        self.criterion = nn.NLLLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.frame_model.parameters()) + list(self.flow_model.parameters()), lr=lr, weight_decay=weight_decay
        )

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        frame, flow_frames = x
        frame_out = self.frame_model(frame)  # logits
        flow_out = self.flow_model(flow_frames)  # logits

        # Convert logits to probabilities along class dimension
        frame_probs = F.softmax(frame_out, dim=1)
        flow_probs = F.softmax(flow_out, dim=1)

        # Average probabilities for late fusion
        return (frame_probs + flow_probs) / 2

    def fit(
        self, num_epochs: int, train_loader: DataLoader, test_loader: DataLoader
    ) -> tuple[list[float], list[float]]:
        """Train the model on the training set and evaluate on validation set.

        Similar to VideoClassifier.fit but expects frames to be stacked along channel dimension.
        """
        train_acc, test_acc = [], []
        self.frame_model.to(self.frame_model.device)
        self.flow_model.to(self.flow_model.device)

        for _ in range(num_epochs):
            # Training phase
            self.frame_model.train()
            self.flow_model.train()
            train_correct = 0
            total_train_samples = 0

            for (frame, flows), target in train_loader:
                # Train flow model
                flows_gpu, target_gpu = flows.to(self.flow_model.device), target.to(self.flow_model.device)
                frame_gpu = frame.to(self.frame_model.device)

                self.optimizer.zero_grad()

                # self.flow_model.optimizer.zero_grad()
                flow_output = self.flow_model(flows_gpu)

                # Train frame model
                # self.frame_model.optimizer.zero_grad()
                frame_output = self.frame_model(frame_gpu)

                # Combine predictions
                avg_output = (F.log_softmax(flow_output, dim=1) + F.log_softmax(frame_output, dim=1)) / 2

                # Compute losses
                flow_loss = self.criterion(flow_output, target_gpu)
                frame_loss = self.criterion(frame_output, target_gpu)
                avg_loss = self.criterion(avg_output, target_gpu)

                # combined_loss = flow_loss + frame_loss
                # combined_loss.backward()

                loss = 0.5 * avg_loss + 0.25 * (flow_loss + frame_loss)
                loss.backward()

                # Update models
                # self.flow_model.optimizer.step()
                # self.frame_model.optimizer.step()
                self.optimizer.step()

                # Compute training accuracy
                predicted = avg_output.argmax(1)
                train_correct += (predicted == target_gpu).sum().cpu().item()
                total_train_samples += target_gpu.size(0)

            # Validation phase
            self.frame_model.eval()
            self.flow_model.eval()
            test_correct = 0
            total_test_samples = 0

            with torch.no_grad():
                for (frame, flows), target in test_loader:
                    flows_gpu, target_gpu = flows.to(self.flow_model.device), target.to(self.flow_model.device)
                    flow_output = self.flow_model(flows_gpu)

                    frame_gpu = frame.to(self.frame_model.device)
                    frame_output = self.frame_model(frame_gpu)

                    avg_output = (F.softmax(flow_output, dim=1) + F.softmax(frame_output, dim=1)) / 2
                    predicted = avg_output.argmax(1)

                    test_correct += (predicted == target_gpu).sum().cpu().item()
                    total_test_samples += target_gpu.size(0)

            train_acc.append(train_correct / total_train_samples)
            test_acc.append(test_correct / total_test_samples)

        return train_acc, test_acc

    def evaluate(self, data: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Evaluate the model on given data and return softmaxed probabilities."""
        frame, flows = data
        flow_out = self.flow_model.evaluate(flows)  # softmaxed probabilities
        frame_out = self.frame_model.evaluate(frame)  # softmaxed probabilities

        # Average probabilities for late fusion
        return (frame_out + flow_out) / 2

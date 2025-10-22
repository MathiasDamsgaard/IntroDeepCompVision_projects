import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.video import R3D_18_Weights, r3d_18


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the baseline classifier.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, C, H, W]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, n_classes]

        """
        return self.base_model(x.to(self.device))

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


class EarlyFusionCNN(nn.Module):
    """Early fusion CNN model for video classification.

    This model stacks all frames along the channel dimension before feeding them to the CNN.
    For example, with 3-channel frames and 10 frames, the input tensor would have 30 channels.
    """

    def __init__(self, n_classes: int = 10, n_frames: int = 10) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_frames = n_frames

        # Modified ResNet with expanded input channels to accommodate stacked frames
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Modify the first conv layer to handle n_frames * 3 channels instead of just 3
        original_conv1 = self.base_model.conv1
        self.base_model.conv1 = nn.Conv2d(
            in_channels=3 * n_frames,  # 3 channels per frame * number of frames
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size[0],
            stride=original_conv1.stride[0],
            padding=original_conv1.padding[0],
            bias=(original_conv1.bias is not None),
        )

        # Replace the final fully connected layer for our classes
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, n_classes)
        self.base_model.to(self.device)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.base_model.parameters(), lr=1e-3, weight_decay=1e-2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for early fusion model.

        Args:
            x: Tensor of shape [batch_size, n_frames, C, H, W] or [batch_size, C, n_frames, H, W]
               depending on how the frames are stacked

        Returns:
            Tensor of shape [batch_size, n_classes] containing class logits

        """
        # Make sure the input is on the same device as the model
        x = x.to(self.device)

        if x.dim() == 5:  # noqa: PLR2004 # [batch_size, C, n_frames, H, W]
            # Check the shape to determine the dimension ordering
            if x.shape[1] == self.n_frames or x.shape[1] < x.shape[2]:
                # [batch_size, n_frames, C, H, W] format
                batch_size, n_frames, c, h, w = x.shape
                # Reshape to [batch_size, n_frames*C, H, W]
                x = x.view(batch_size, n_frames * c, h, w)
            else:
                # [batch_size, C, n_frames, H, W] format
                batch_size, c, n_frames, h, w = x.shape
                # Reshape to [batch_size, C*n_frames, H, W]
                x = x.view(batch_size, c * n_frames, h, w)

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
                # For FrameImageDataset, each frame is [batch_size, 3, H, W]
                # But EarlyFusionCNN expects [batch_size, n_frames*3, H, W]
                # We'll duplicate the frame to match the expected input size
                data_gpu, target_gpu = data.to(self.device), target.to(self.device)

                # Create a modified input by repeating the same frame n_frames times
                # Create input with the right shape by repeating the frame
                expanded_data = data_gpu.repeat(1, self.n_frames, 1, 1)

                self.optimizer.zero_grad()
                output = self.base_model(expanded_data)
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

                    # Create a modified input by repeating the same frame n_frames times
                    expanded_data = data_gpu.repeat(1, self.n_frames, 1, 1)

                    output = self.base_model(expanded_data)
                    predicted = output.argmax(1)
                    test_correct += (predicted == target_gpu).sum().cpu().item()
                    total_test_samples += target_gpu.size(0)

            train_acc.append(train_correct / total_train_samples)
            test_acc.append(test_correct / total_test_samples)

        return train_acc, test_acc

    def evaluate(self, data: torch.Tensor) -> torch.Tensor:
        """Evaluate the model on given data and return raw logits."""
        self.base_model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            output = self.base_model(data)
        return output.cpu()


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

        # Remove the final classification layer, keeping the feature extractor
        self.feature_size = self.base_model.fc.in_features  # Usually 512 for ResNet18
        # self.base_model.fc = nn.Identity()  # Replace with Identity to get features
        # Use a Linear layer that behaves like Identity (satisfies type checker)
        self.base_model.fc = nn.Linear(self.feature_size, self.feature_size)
        # Initialize as identity matrix
        with torch.no_grad():
            self.base_model.fc.weight.copy_(torch.eye(self.feature_size))
            self.base_model.fc.bias.zero_()

        # Temporal fusion: After getting features from each frame
        self.temporal_fusion = nn.Sequential(
            nn.Linear(self.feature_size, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, n_classes)
        )

        # Freeze backbone except last layer
        for name, param in self.base_model.named_parameters():
            if "layer4" not in name:  # Only fine-tune last conv block
                param.requires_grad = False

        # Move to device
        self.base_model.to(self.device)
        self.temporal_fusion.to(self.device)

        # Optimizer and loss
        trainable_params = list(self.temporal_fusion.parameters()) + list(
            filter(lambda p: p.requires_grad, self.base_model.parameters())
        )
        self.optimizer = torch.optim.Adam(trainable_params, lr=1e-3, weight_decay=1e-2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass for late fusion model.

        Args:
            x: List of frame tensors, each of shape [batch_size, C, H, W]

        Returns:
            Tensor of shape [batch_size, n_classes] containing class logits

        """
        _batch_size = x[0].size(0)

        # Extract features from each frame
        frame_features = []
        for frame in x:
            features = self.base_model(frame.to(self.device))  # [batch_size, feature_size]
            frame_features.append(features)

        # Average the features across frames
        avg_features = torch.stack(frame_features).mean(dim=0)  # [batch_size, feature_size]

        # Final classification
        return self.temporal_fusion(avg_features)

    def fit(
        self, num_epochs: int, train_loader: DataLoader, test_loader: DataLoader
    ) -> tuple[list[float], list[float]]:
        """Train the model on the training set and evaluate on validation set."""
        train_acc, test_acc = [], []

        for _ in range(num_epochs):
            # Training phase
            self.base_model.train()
            self.temporal_fusion.train()
            train_correct = 0
            total_train_samples = 0

            for data, targets in train_loader:
                # For FrameImageDataset, each item is a single frame, not a list of frames
                # So we need to handle it differently than in the forward() method
                targets_gpu = targets.to(self.device)
                data_gpu = data.to(self.device)

                self.optimizer.zero_grad()

                # Process single frame directly with base_model
                features = self.base_model(data_gpu)
                outputs = self.temporal_fusion(features)

                loss = self.criterion(outputs, targets_gpu)
                loss.backward()
                self.optimizer.step()

                predicted = outputs.argmax(1)
                train_correct += (predicted == targets_gpu).sum().cpu().item()
                total_train_samples += targets.size(0)

            # Validation phase
            self.base_model.eval()
            self.temporal_fusion.eval()
            test_correct = 0
            total_test_samples = 0

            with torch.no_grad():
                for data, targets in test_loader:
                    data_gpu = data.to(self.device)
                    targets_gpu = targets.to(self.device)

                    # Process single frame directly with base_model
                    features = self.base_model(data_gpu)
                    outputs = self.temporal_fusion(features)

                    predicted = outputs.argmax(1)
                    test_correct += (predicted == targets_gpu).sum().cpu().item()
                    total_test_samples += targets.size(0)

            train_acc.append(train_correct / total_train_samples)
            test_acc.append(test_correct / total_test_samples)

        return train_acc, test_acc

    def evaluate(self, data: list[torch.Tensor]) -> torch.Tensor:
        """Evaluate the model on given data and return raw logits."""
        self.base_model.eval()
        self.temporal_fusion.eval()
        with torch.no_grad():
            output = self.forward(data)
        return output.cpu()


class CNN3D(nn.Module):
    """3D CNN model for video classification.

    This model uses 3D convolutions (ResNet-18 backbone) to jointly capture
    spatial and temporal information in videos.
    """

    def __init__(self, n_classes: int = 10, n_frames: int = 10) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_frames = n_frames
        self.n_classes = n_classes

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
        """Evaluate the model on given data and return raw logits."""
        self.base_model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            output = self.base_model(data)
        return output.cpu()


class FlowCNN(nn.Module):
    def __init__(self, n_classes: int = 10, lr: float = 0.001, weight_decay: float = 0) -> None:
        super().__init__()
        self.frame_model = BaselineClassifier(n_classes=n_classes)
        self.flow_model = EarlyFusionCNN(
            n_classes=n_classes, n_frames=6
        )  # (((n_frames - 1) * 2)//3) because we have EarlyFusion expects input to be 3 channels for 10 frames,
        # but we have 2 channels for 9 frames
        self.frame_model.optimizer = torch.optim.Adam(self.frame_model.parameters(), lr=lr, weight_decay=weight_decay)
        self.flow_model.optimizer = torch.optim.Adam(self.flow_model.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        frame, flow_frames = x
        frame_out = self.frame_model(frame)
        flow_out = self.flow_model(flow_frames)
        return (frame_out + flow_out) / 2

    def fit(
        self, num_epochs: int, train_loader: DataLoader, test_loader: DataLoader
    ) -> tuple[list[float], list[float]]:
        """Train the model on the training set and evaluate on validation set.

        Similar to VideoClassifier.fit but expects frames to be stacked along channel dimension.
        """
        train_acc, test_acc = [], []
        self.frame_model.to(self.device)
        self.flow_model.to(self.device)

        for _ in range(num_epochs):
            # Training phase
            self.frame_model.train()
            self.flow_model.train()
            train_correct = 0
            total_train_samples = 0

            for (frame, flow), target in train_loader:
                flow_gpu, target_gpu = flow.to(self.device), target.to(self.device)

                expanded_flow = flow_gpu.repeat(1, self.n_frames, 1, 1)

                self.flow_model.optimizer.zero_grad()
                flow_output = self.flow_model(expanded_flow)
                flow_loss = self.flow_model.criterion(flow_output, target_gpu)
                flow_loss.backward()
                self.flow_model.optimizer.step()

                frame_gpu = frame.to(self.device)
                self.frame_model.optimizer.zero_grad()
                frame_output = self.frame_model(frame_gpu)
                frame_loss = self.frame_model.criterion(frame_output, target_gpu)
                frame_loss.backward()
                self.frame_model.optimizer.step()

                avg_output = (flow_output + frame_output) / 2
                predicted = avg_output.argmax(1)

                train_correct += (predicted == target_gpu).sum().cpu().item()
                total_train_samples += target_gpu.size(0)

            # Validation phase
            self.frame_model.eval()
            self.flow_model.eval()
            test_correct = 0
            total_test_samples = 0

            with torch.no_grad():
                for (frame, flow), target in test_loader:
                    flow_gpu, target_gpu = flow.to(self.device), target.to(self.device)
                    expanded_flow = flow_gpu.repeat(1, self.n_frames, 1, 1)
                    flow_output = self.flow_model(expanded_flow)

                    frame_gpu = frame.to(self.device)
                    frame_output = self.frame_model(frame_gpu)

                    avg_output = (flow_output + frame_output) / 2
                    predicted = avg_output.argmax(1)

                    test_correct += (predicted == target_gpu).sum().cpu().item()
                    total_test_samples += target_gpu.size(0)

            train_acc.append(train_correct / total_train_samples)
            test_acc.append(test_correct / total_test_samples)

        return train_acc, test_acc

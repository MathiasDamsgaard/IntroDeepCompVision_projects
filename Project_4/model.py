from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def get_model(num_classes: int = 2) -> nn.Module:
    """Create a ResNet18 model for classification.

    Args:
        num_classes (int): Number of output classes. Defaults to 2.

    Returns:
        nn.Module: The model.

    """
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Freeze backbone (all layers except the final fc layer)
    # Only train the final classification layer
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False

    return model

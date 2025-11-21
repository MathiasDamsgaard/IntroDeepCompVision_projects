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

    # Freeze all parameters in the backbone
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

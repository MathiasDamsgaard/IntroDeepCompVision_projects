import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    """Binary Cross Entropy Loss for binary segmentation."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Numerically stable BCE loss
        # Uses torch.abs to ensure exp(-abs(y_pred)) is always <= 1, preventing overflow
        # torch.log1p provides better precision for log(1 + x) when x is small
        return torch.mean(torch.clamp(y_pred, min=0) - y_true * y_pred + torch.log1p(torch.exp(-torch.abs(y_pred))))


class FocalLoss(nn.Module):
    """Focal Loss - focuses on hard examples by down-weighting easy ones."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(y_pred)

        # Focal loss formula
        # For positive class (y_true = 1): -alpha * (1-p)^gamma * log(p)
        # For negative class (y_true = 0): -(1-alpha) * p^gamma * log(1-p)
        ce_loss = -(y_true * torch.log(p + 1e-8) + (1 - y_true) * torch.log(1 - p + 1e-8))
        p_t = y_true * p + (1 - y_true) * (1 - p)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss
        return torch.mean(loss)


class WeightedCrossEntropyLoss(nn.Module):
    """Binary Cross Entropy Loss with positive class weighting for handling class imbalance."""

    def __init__(self, pos_weight: float | None = None) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # If pos_weight not set, calculate it from the batch
        if self.pos_weight is None:
            # Calculate ratio of negative to positive samples
            num_pos = torch.sum(y_true)
            num_neg = torch.sum(1 - y_true)
            pos_weight = num_neg / (num_pos + 1e-8)
        else:
            pos_weight = self.pos_weight

        # Numerically stable BCE loss with positive weights
        # pos_weight multiplies the loss for positive class
        max_val = torch.clamp(y_pred, min=0)
        loss = max_val - y_true * y_pred + torch.log1p(torch.exp(-torch.abs(y_pred)))

        # Apply weight to positive class
        weighted_loss = (1 - y_true) * loss + y_true * pos_weight * loss

        return torch.mean(weighted_loss)

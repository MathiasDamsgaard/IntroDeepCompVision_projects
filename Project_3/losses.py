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
    """Focal Loss - focuses on hard examples by down-weighting easy ones.

    Uses gamma=2.0 as proposed by the authors and calculates alpha as inverse class frequency.
    """

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Calculate alpha as inverse class frequency from the batch
        num_pos = torch.sum(y_true)
        num_neg = torch.sum(1 - y_true)
        alpha = num_neg / (num_pos + num_neg + 1e-8)  # Weight for positive class

        # Numerically stable BCE loss (same formula as CrossEntropyLoss)
        ce_loss = torch.clamp(y_pred, min=0) - y_true * y_pred + torch.log1p(torch.exp(-torch.abs(y_pred)))

        # Calculate probability for focal weighting
        p = torch.sigmoid(y_pred)
        p_t = y_true * p + (1 - y_true) * (1 - p)

        # Apply focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting: alpha for positive class, (1-alpha) for negative class
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)

        # Focal Loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
        # The negative sign is implicit in our ce_loss which is already -log(p_t)
        loss = alpha_t * focal_weight * ce_loss
        return torch.mean(loss)


class WeightedCrossEntropyLoss(nn.Module):
    """Binary Cross Entropy Loss with positive class weighting for handling class imbalance.

    Automatically calculates pos_weight as the ratio of negative to positive samples from the data.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Calculate ratio of negative to positive samples from the batch
        num_pos = torch.sum(y_true)
        num_neg = torch.sum(1 - y_true)
        pos_weight = num_neg / (num_pos + 1e-8)

        # Numerically stable BCE loss (same formula as CrossEntropyLoss)
        loss = torch.clamp(y_pred, min=0) - y_true * y_pred + torch.log1p(torch.exp(-torch.abs(y_pred)))

        # Apply weight to positive class
        weighted_loss = (1 - y_true) * loss + y_true * pos_weight * loss

        return torch.mean(weighted_loss)


class PointBCELoss(nn.Module):
    """Binary Cross Entropy loss that is only calculated at specific point locations."""

    def __init__(self) -> None:
        super().__init__()
        # Use BCEWithLogitsLoss because it's numerically stable and takes raw model outputs (logits)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, y_pred: torch.Tensor, y_clicks: torch.Tensor) -> torch.Tensor:
        """Calculate BCE loss only at the locations specified by y_clicks.

        Args:
            y_pred (torch.Tensor): The model's output prediction map. Shape [B, 1, H, W].
            y_clicks (torch.Tensor): The click mask with 1s, 0s, and ignore values. Shape [B, 1, H, W].

        """
        # Find the locations of the clicks (where the mask is not the ignore value)
        active_criteria = -100
        active_pixels = y_clicks != active_criteria

        # If there are no clicks in the batch, return a loss of 0
        if not active_pixels.any():
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        # Select only the predictions and labels at the click locations
        pred_clicks = y_pred[active_pixels]
        true_clicks = y_clicks[active_pixels]

        # Calculate the BCE loss on only these points
        return self.bce_loss(pred_clicks, true_clicks.float())

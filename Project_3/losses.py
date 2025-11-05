import torch
from torch import nn


class BCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred, y_true):
        # Numerically stable BCE loss
        # Uses torch.abs to ensure exp(-abs(y_pred)) is always <= 1, preventing overflow
        # torch.log1p provides better precision for log(1 + x) when x is small
        return torch.mean(torch.clamp(y_pred, min=0) - y_true * y_pred + torch.log1p(torch.exp(-torch.abs(y_pred))))


class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred, y_true):
        return 1 - (torch.mean(2 * y_true * y_pred + 1) / (torch.mean(y_true + y_pred) + 1))


class FocalLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred, y_true):
        gamma = 2.0
        return -torch.sum(
            (1 - torch.sigmoid(y_pred)) ** gamma * y_true * torch.log(torch.sigmoid(y_pred))
            + (1 - y_true) * torch.log(1 - torch.sigmoid(y_pred) ** gamma)
        )


class BCELoss_TotalVariation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true * y_pred + torch.log(1 + torch.exp(-y_pred)))
        regularization = torch.sigmoid(y_pred)  # sparsity term
        return loss + 0.1 * regularization

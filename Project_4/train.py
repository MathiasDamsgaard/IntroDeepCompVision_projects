import argparse
from pathlib import Path

import torch
from dataset import get_dataloader
from logger import logger
from model import get_model
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): The data loader.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        device (torch.device): The device to use.

    Returns:
        tuple[float, float]: The average loss and accuracy.

    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training"):
        images_dev, labels_dev = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images_dev)
        loss = criterion(outputs, labels_dev)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images_dev.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels_dev.size(0)
        correct += (predicted == labels_dev).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The data loader.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to use.

    Returns:
        tuple[float, float]: The average loss and accuracy.

    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images_dev, labels_dev = images.to(device), labels.to(device)

            outputs = model(images_dev)
            loss = criterion(outputs, labels_dev)

            running_loss += loss.item() * images_dev.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels_dev.size(0)
            correct += (predicted == labels_dev).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main() -> None:
    """Run the training and evaluation pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Paths
    script_dir = Path(__file__).parent
    image_dir = "/dtu/datasets1/02516/potholes/images/"
    train_pickle = script_dir / "proposals_data/train_proposals.pkl"
    val_pickle = script_dir / "proposals_data/val_proposals.pkl"
    test_pickle = script_dir / "proposals_data/test_proposals.pkl"

    # DataLoaders
    train_loader = get_dataloader(str(train_pickle), image_dir, batch_size=args.batch_size, is_train=True)
    val_loader = get_dataloader(str(val_pickle), image_dir, batch_size=args.batch_size, is_train=False)
    test_loader = get_dataloader(str(test_pickle), image_dir, batch_size=args.batch_size, is_train=False)

    # Model
    model = get_model(num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save last model
    torch.save(model.state_dict(), output_dir / "last_model.pth")
    logger.info("Saved last model")

    # Test
    logger.info("Training complete. Starting testing...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()

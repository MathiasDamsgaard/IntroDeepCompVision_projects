import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from dataset.drive_dataset import Drive
from dataset.ph2_dataset import Ph2
from losses import CrossEntropyLoss, FocalLoss, WeightedCrossEntropyLoss
from model.encdec_model import EncDec
from model.unet_model import UNet
from torch import nn, optim
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms
from tqdm import tqdm

# Global seed for reproducibility (must match predict.py)
RANDOM_SEED = 42

# Constants for determining x-axis tick spacing
SHORT_TRAINING_THRESHOLD = 20
MEDIUM_TRAINING_THRESHOLD = 50


def create_data_loaders(
    dataset_name: str,
    size: int,
    batch_size: int,
    workers: int,
    test_split: float,
    val_split: float,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders with proper splits.

    Args:
        dataset_name: Name of dataset ("Drive" or "Ph2")
        size: Image size for resizing
        batch_size: Batch size for data loaders
        workers: Number of worker threads for data loading
        test_split: Ratio for test set split
        val_split: Ratio for validation set split (applied after test split)

    Returns:
        Tuple of (train_loader, val_loader)

    """
    # Define transforms
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    augmentation = transforms.Compose(
        [transforms.Resize((size, size)), transforms.RandomRotation(10), transforms.ToTensor()]
    )

    # Load dataset based on choice - create separate datasets for train and val
    if dataset_name == "Drive":
        # Create dataset WITH augmentation for training
        train_dataset_full = Drive(transform=augmentation)
        # Create dataset WITHOUT augmentation for validation
        val_dataset_full = Drive(transform=transform)
    elif dataset_name == "Ph2":
        # Ph2 doesn't use augmentation
        train_dataset_full = Ph2(transform=transform)
        val_dataset_full = Ph2(transform=transform)
    else:
        msg = f"Unknown dataset: {dataset_name}"
        raise ValueError(msg)

    # Create train/val/test splits - get indices
    total_size = len(train_dataset_full)

    # First split: separate test set
    test_size = int(test_split * total_size)
    train_val_size = total_size - test_size

    # Get indices for splits
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_val_indices = torch.randperm(total_size, generator=generator).tolist()[:train_val_size]

    # Second split: separate validation from training
    val_size = int(val_split * train_val_size)
    train_size = train_val_size - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    shuffled = torch.randperm(train_val_size, generator=generator).tolist()
    train_indices = [train_val_indices[i] for i in shuffled[:train_size]]
    val_indices = [train_val_indices[i] for i in shuffled[train_size:]]

    # Create subsets using the appropriate datasets
    train_subset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset_full, val_indices)

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=workers)

    return train_loader, val_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int,
    device: torch.device,
) -> tuple[list, list]:
    """Train the model and return training history.

    Args:
        model: Neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        loss_fn: Loss function
        optimizer: Optimizer
        epochs: Number of training epochs
        device: Device to train on (cuda/cpu)

    Returns:
        Tuple of (train_losses, val_losses) lists

    """
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs), desc="Training"):
        # Training phase
        model.train()
        train_loss = 0
        for x_batch, y_true in train_loader:
            x_batch_gpu = x_batch.to(device)
            y_true_gpu = y_true.to(device)

            # Set parameter gradients to zero
            optimizer.zero_grad()

            # Forward + backward + optimize
            y_pred = model(x_batch_gpu)
            loss = loss_fn(y_pred, y_true_gpu)
            loss.backward()
            optimizer.step()

            # Accumulate loss
            train_loss += loss.item() / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_true in val_loader:
                x_batch_gpu = x_batch.to(device)
                y_true_gpu = y_true.to(device)

                y_pred = model(x_batch_gpu)
                loss = loss_fn(y_pred, y_true_gpu)
                val_loss += loss.item() / len(val_loader)

        # Store losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print epoch summary
        if (epoch + 1) % 5 == 0 or epoch == 0:
            tqdm.write(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses


def save_results(
    model: nn.Module,
    train_losses: list,
    val_losses: list,
    dataset_name: str,
    model_name: str,
    loss_name: str,
    epochs: int,
    output_dir: str,
) -> None:
    """Save model checkpoint and training plots.

    Args:
        model: Trained model
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        dataset_name: Name of dataset
        model_name: Name of model architecture
        loss_name: Name of loss function
        epochs: Number of epochs trained
        output_dir: Directory to save outputs

    """
    # Create filename base
    filename_base = f"{dataset_name.lower()}_{model_name.lower()}_{loss_name.lower()}"

    # Save model checkpoint
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_filename = checkpoint_dir / f"{filename_base}.pth"
    torch.save(model.state_dict(), model_filename)

    # Create and save loss plot
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, train_losses, label="Train Loss", marker="o", linewidth=2)
    plt.plot(epochs_range, val_losses, label="Validation Loss", marker="s", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"Training History: {dataset_name} - {model_name} - {loss_name}", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(visible=True, alpha=0.3)

    # Set integer ticks on x-axis
    if epochs <= SHORT_TRAINING_THRESHOLD:
        plt.xticks(epochs_range)
    else:
        # For longer training, use step size of 5 or 10
        step = 5 if epochs <= MEDIUM_TRAINING_THRESHOLD else 10
        plt.xticks(range(1, epochs + 1, step))

    plt.tight_layout()

    plot_filename = figures_dir / f"{filename_base}_loss.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Run model training pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument(
        "--dataset", type=str, default="Drive", choices=["Drive", "Ph2"], help="Dataset to use for training"
    )
    parser.add_argument(
        "--model", type=str, default="EncDec", choices=["EncDec", "UNet"], help="Model architecture to use"
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="CrossEntropyLoss",
        choices=["CrossEntropyLoss", "FocalLoss", "WeightedCrossEntropyLoss"],
        help="Loss function to use",
    )
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--size", type=int, default=128, help="Image size (images will be resized to size x size)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--workers", type=int, default=3, help="Number of data loading workers")
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=None,
        help="Positive class weight for WeightedCrossEntropyLoss (auto-calculated if not provided)",
    )
    parser.add_argument("--test_split", type=float, default=0.10, help="Test split ratio")
    parser.add_argument("--val_split", type=float, default=0.10, help="Validation split ratio applied after test split")
    parser.add_argument("--output_dir", type=str, default="model", help="Directory to save model outputs and results")
    args = parser.parse_args()

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        dataset_name=args.dataset,
        size=args.size,
        batch_size=args.batch_size,
        workers=args.workers,
        test_split=args.test_split,
        val_split=args.val_split,
    )

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {
        "EncDec": EncDec,
        "UNet": UNet,
    }
    model = models[args.model]().to(device)
    summary(model, (3, args.size, args.size))

    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), args.lr)

    loss_fns = {
        "CrossEntropyLoss": CrossEntropyLoss,
        "FocalLoss": FocalLoss,
        "WeightedCrossEntropyLoss": WeightedCrossEntropyLoss,
    }

    if args.loss == "WeightedCrossEntropyLoss":
        loss_fn = loss_fns[args.loss](pos_weight=args.pos_weight)
    else:
        loss_fn = loss_fns[args.loss]()

    # Train model
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
    )

    # Save results
    save_results(
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        dataset_name=args.dataset,
        model_name=args.model,
        loss_name=args.loss,
        epochs=args.epochs,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

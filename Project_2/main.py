import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import FrameImageDataset, FrameVideoDataset
from logger import logger
from model import CNN3D, BaselineClassifier, EarlyFusionCNN, LateFusionCNN
from scipy import stats
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# Define a type for our models
ModelType = BaselineClassifier | LateFusionCNN | EarlyFusionCNN | CNN3D


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model training and evaluation."""
    parser = argparse.ArgumentParser(description="Train and evaluate video classifiers on UFC10 dataset")

    # Dataset parameters
    parser.add_argument(
        "--root_dir", type=str, default="/dtu/datasets1/02516/ufc10", help="Root directory of the dataset"
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["2D_CNN_aggr", "2D_CNN_late_fusion", "2D_CNN_early_fusion", "3D_CNN", "all"],
        help="Model architecture to use, or 'all' to run all models",
    )

    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for optimizer")
    parser.add_argument("--image_size", type=int, default=64, help="Image size for resizing (square)")

    # Cross-validation parameters
    parser.add_argument("--cross_validate", action="store_true", help="Whether to perform cross-validation")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of cross-validation folds")

    # Output and checkpointing
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save model outputs")
    parser.add_argument("--save_model", action="store_true", help="Whether to save the trained model")
    parser.add_argument("--no_leakage", action="store_true", help="Whether to use no-leakage dataset version")

    return parser.parse_args()


def create_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for training, validation, and testing.

    For 3D_CNN, we use FrameVideoDataset with stack_frames=True to get [C, T, H, W] per sample.
    For 2D models, we keep training/val on individual frames and test on lists of frames.
    """
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()])

    root_path = Path(args.root_dir)
    root_path = root_path / "ucf101_noleakage" if args.no_leakage else root_path / "ufc10"
    root_dir = str(root_path)

    if args.model == "3D_CNN":
        # Use video-level samples with frames stacked: [C, T, H, W]
        train_ds = FrameVideoDataset(root_dir=root_dir, split="train", transform=transform, stack_frames=True)
        val_ds = FrameVideoDataset(root_dir=root_dir, split="val", transform=transform, stack_frames=True)
        test_ds = FrameVideoDataset(root_dir=root_dir, split="test", transform=transform, stack_frames=True)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

    else:
        # 2D models: frame-level training/val, video-level testing as list of frames
        frameimagetrain_dataset = FrameImageDataset(root_dir=root_dir, split="train", transform=transform)
        frameimageval_dataset = FrameImageDataset(root_dir=root_dir, split="val", transform=transform)
        framevideotest_dataset = FrameVideoDataset(
            root_dir=root_dir, split="test", transform=transform, stack_frames=False
        )

        train_loader = DataLoader(frameimagetrain_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(frameimageval_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(framevideotest_dataset, batch_size=args.batch_size, shuffle=True)

    logger.info(f"Created dataloaders with batch size {args.batch_size}")
    return train_loader, val_loader, test_loader


def create_model(args: argparse.Namespace) -> BaselineClassifier | LateFusionCNN | EarlyFusionCNN | CNN3D:
    """Create model based on the specified architecture."""
    n_classes = 10  # UFC10 dataset has 10 classes
    n_frames = 10  # Number of frames per video

    if args.model == "2D_CNN_aggr":
        # Standard 2D CNN with frame aggregation during evaluation
        model = BaselineClassifier(n_classes=n_classes)
        model.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.base_model.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    elif args.model == "2D_CNN_late_fusion":
        # Late fusion model processes frames independently then combines
        model = LateFusionCNN(n_classes=n_classes)
        model.optimizer = torch.optim.Adam(
            list(model.temporal_fusion.parameters())
            + list(filter(lambda p: p.requires_grad, model.base_model.parameters())),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    elif args.model == "2D_CNN_early_fusion":
        # Early fusion model stacks frames along channel dimension
        model = EarlyFusionCNN(n_classes=n_classes, n_frames=n_frames)
        model.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    elif args.model == "3D_CNN":
        # 3D CNN model
        model = CNN3D(n_classes=n_classes, n_frames=n_frames)
        model.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    else:
        msg = f"Unknown model architecture: {args.model}"
        raise ValueError(msg)

    logger.info(f"Created {args.model} model")
    return model


def evaluate_on_videos(
    model: ModelType,
    test_loader: DataLoader,
    model_name: str,
) -> float:
    """Evaluate the model on complete test videos."""
    logger.info("Evaluating model on test videos...")
    n_correct = 0
    n_frames = 10  # Number of frames per video
    total_samples = 0

    # Expected tensor dimensions
    tensor_channel_size = 4
    tensor_batch_size = 5

    for frames, targets in test_loader:
        batch_size = targets.size(0)
        total_samples += batch_size

        # Make frames a tensor with shape [B, C, T, H, W] when needed
        frames_tensor = torch.stack(frames, dim=2) if isinstance(frames, list) else frames
        if frames_tensor.dim() == tensor_channel_size:
            # If dataset provided [C, T, H, W], add batch dim
            frames_tensor = frames_tensor.unsqueeze(0)
        if frames_tensor.dim() != tensor_batch_size:
            msg = f"Expected frames with 5 dims [B, C, T, H, W], got {frames_tensor.shape}"
            raise ValueError(msg)

        batch_size, c, n_frames, h, w = frames_tensor.shape

        # Handle different model architectures
        device = next(model.parameters()).device

        if model_name == "2D_CNN_aggr":
            # Process each frame independently and average predictions
            predictions = torch.zeros((batch_size, 10)).to(device)

            for i in range(n_frames):
                # Extract frame i from all videos in batch
                batch = frames_tensor[:, :, i, :, :].to(device)
                with torch.no_grad():
                    logits = model(batch)
                predictions += torch.softmax(logits, dim=1)

            predictions /= n_frames  # Average over number of frames

        elif model_name == "2D_CNN_late_fusion":
            # Late fusion model needs a list of frame tensors
            # Use list comprehension to create a list of frame tensors
            frame_list = [frames_tensor[:, :, i, :, :] for i in range(n_frames)]

            with torch.no_grad():
                logits = model(frame_list)
            predictions = torch.softmax(logits, dim=1)

        elif model_name == "2D_CNN_early_fusion":
            # Early fusion expects all frames stacked in channels
            # Reshape from [batch, C, n_frames, H, W] to [batch, C*n_frames, H, W]
            batch_size, c, n_frames, h, w = frames_tensor.shape
            stacked_frames = frames_tensor.reshape(batch_size, c * n_frames, h, w)
            # Make sure the stacked_frames are on the same device as the model
            stacked_frames = stacked_frames.to(device)
            with torch.no_grad():
                logits = model(stacked_frames)
            predictions = torch.softmax(logits, dim=1)

        elif model_name == "3D_CNN":
            # For 3D CNNs, pass the whole clip [B, C, T, H, W]
            with torch.no_grad():
                logits = model(frames_tensor.to(device))
            predictions = torch.softmax(logits, dim=1)

        else:
            msg = f"Evaluation not implemented for model: {model_name}"
            raise NotImplementedError(msg)

        # Move targets to the same device as predictions for comparison
        targets_device = targets.to(predictions.device)

        # Count correct predictions
        n_correct += (predictions.argmax(1) == targets_device).sum().item()

    # Calculate and return accuracy
    accuracy = n_correct / total_samples * 100
    logger.info(f"Test accuracy on complete videos: {accuracy:.2f}%")
    return accuracy


def save_results(
    model: ModelType,
    train_acc: list,
    val_acc: list,
    test_acc: float,
    args: argparse.Namespace,
) -> None:
    """Save model and results."""
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save model if requested
    if args.save_model and args.no_leakage:
        model_path = output_dir / f"{args.model}_model_noleak.pt"
    elif args.save_model and not args.no_leakage:
        model_path = output_dir / f"{args.model}_model.pt"

    if args.save_model:
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

    # Save training history
    results = {
        "model": args.model,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "args": vars(args),
    }
    if args.no_leakage:
        results_path = output_dir / f"{args.model}_results_noleak.pkl"
    else:
        results_path = output_dir / f"{args.model}_results.pkl"
    with Path(results_path).open("wb") as f:
        pickle.dump(results, f)
    logger.info(f"Results saved to {results_path}")

    # Plot accuracy curves
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_acc) + 1)
    plt.plot(epochs, train_acc, label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.axhline(y=test_acc / 100, color="r", linestyle="-", label=f"Test Accuracy: {test_acc:.2f}%")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    if args.no_leakage:
        plt.title(f"Model: {args.model} with no leakage")
    else:
        plt.title(f"Model: {args.model} with leakage")
    plt.legend()
    plt.grid(visible=True)
    plt.xticks(epochs)
    if args.no_leakage:
        plt_path = output_dir / f"{args.model}_accuracy_noleak.png"
    else:
        plt_path = output_dir / f"{args.model}_accuracy.png"
    plt.tight_layout()
    plt.savefig(plt_path)
    logger.info(f"Accuracy plot saved to {plt_path}")


def create_fold_dataloaders(
    root_dir: str,
    model_name: str,
    image_size: int,
    batch_size: int,
    n_folds: int = 5,
) -> tuple[list[DataLoader], list[DataLoader], DataLoader]:
    """Create training and validation dataloaders for k-fold cross validation."""
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

    # Create datasets based on model type
    if model_name == "3D_CNN":
        full_dataset = FrameVideoDataset(root_dir=root_dir, split="train", transform=transform, stack_frames=True)
        test_dataset = FrameVideoDataset(root_dir=root_dir, split="test", transform=transform, stack_frames=True)
    else:
        full_dataset = FrameImageDataset(root_dir=root_dir, split="train", transform=transform)
        test_dataset = FrameVideoDataset(root_dir=root_dir, split="test", transform=transform, stack_frames=False)

    # Create indices for k-fold CV
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Create train and validation loaders for each fold
    train_loaders = []
    val_loaders = []

    # Get indices for each fold
    indices = list(range(len(full_dataset)))
    for train_idx, val_idx in kfold.split(indices):  # pyright: ignore[reportArgumentType]
        train_subset = Subset(full_dataset, train_idx)  # pyright: ignore[reportArgumentType]
        val_subset = Subset(full_dataset, val_idx)  # pyright: ignore[reportArgumentType]

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    # Create a single test loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loaders, val_loaders, test_loader


def compute_confidence_intervals(accuracies: list[list[float]]) -> tuple[list[float], list[float], list[float]]:
    """Compute mean and 95% confidence intervals for accuracies across folds."""
    # Transpose to get lists by epoch
    n_epochs = len(accuracies[0])
    n_folds = len(accuracies)

    # Compute means and confidence intervals
    means = []
    lower_bounds = []
    upper_bounds = []

    for epoch in range(n_epochs):
        epoch_accs = [accuracies[fold][epoch] for fold in range(n_folds)]
        mean = np.mean(epoch_accs)
        std = np.std(epoch_accs)

        # 95% CI with t-distribution (small sample size)
        ci = stats.t.interval(0.95, len(epoch_accs) - 1, loc=mean, scale=std / np.sqrt(len(epoch_accs)))

        means.append(mean)
        lower_bounds.append(ci[0])
        upper_bounds.append(ci[1])

    return means, lower_bounds, upper_bounds


def plot_cv_results(
    train_accs: list[list[float]],
    val_accs: list[list[float]],
    test_accs: list[float],
    model_name: str,
    output_dir: Path,
) -> None:
    """Plot cross-validation results with confidence intervals."""
    # Compute confidence intervals
    train_means, train_lower, train_upper = compute_confidence_intervals(train_accs)
    val_means, val_lower, val_upper = compute_confidence_intervals(val_accs)

    test_mean: float = np.mean(test_accs)  # pyright: ignore[reportAssignmentType]
    test_std: float = np.std(test_accs)  # pyright: ignore[reportAssignmentType]
    test_ci: tuple[float, float] = stats.t.interval(
        0.95, len(test_accs) - 1, loc=test_mean, scale=test_std / np.sqrt(len(test_accs))
    )

    # Create plot
    plt.figure(figsize=(12, 8))

    # Plot training accuracy
    epochs = range(1, len(train_means) + 1)
    plt.plot(epochs, train_means, "b-", label="Training Accuracy")
    plt.fill_between(epochs, train_lower, train_upper, color="b", alpha=0.2)

    # Plot validation accuracy
    plt.plot(epochs, val_means, "r-", label="Validation Accuracy")
    plt.fill_between(epochs, val_lower, val_upper, color="r", alpha=0.2)

    # Plot test accuracy
    plt.axhline(
        y=test_mean / 100,
        color="g",
        linestyle="-",
        label=f"Test Accuracy: {test_mean:.2f}% ± {(test_ci[1] - test_mean):.2f}%",
    )
    plt.axhspan(test_ci[0] / 100, test_ci[1] / 100, color="g", alpha=0.2)

    # Add labels and title
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"5-Fold Cross Validation Results: {model_name}")
    plt.legend()
    plt.grid(visible=True)

    # Save plot
    plot_path = output_dir / f"{model_name}_cv_plot.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved CV results plot to {plot_path}")


def run_cross_validation(args: argparse.Namespace, model_name: str) -> None:
    """Run cross-validation for a specific model."""
    logger.info(f"Running {args.n_folds}-fold cross-validation for {model_name}")

    # Create dataloaders for cross-validation
    train_loaders, val_loaders, test_loader = create_fold_dataloaders(
        root_dir=args.root_dir,
        model_name=model_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        n_folds=args.n_folds,
    )

    # Lists to store results for each fold
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []

    # Run cross-validation
    for fold in range(args.n_folds):
        logger.info(f"Starting fold {fold + 1}/{args.n_folds}")

        # Create model
        model = create_model(args)

        # Train model
        train_acc, val_acc = model.fit(
            num_epochs=args.num_epochs,
            train_loader=train_loaders[fold],
            test_loader=val_loaders[fold],
        )

        # Test model
        test_acc = evaluate_on_videos(model, test_loader, model_name)

        # Save results
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)

        logger.info(f"Fold {fold + 1} results:")
        logger.info(f"  Train accuracy: {train_acc[-1]:.4f}")
        logger.info(f"  Validation accuracy: {val_acc[-1]:.4f}")
        logger.info(f"  Test accuracy: {test_acc:.4f}")

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save results dictionary
    results = {
        "model": model_name,
        "train_accs": train_accuracies,
        "val_accs": val_accuracies,
        "test_accs": test_accuracies,
        "args": vars(args),
    }

    # Save results to file
    results_path = output_dir / f"{model_name}_cv_results.pkl"
    with Path.open(results_path, "wb") as f:
        pickle.dump(results, f)

    logger.info(f"Saved CV results to {results_path}")

    # Plot cross-validation results
    plot_cv_results(train_accuracies, val_accuracies, test_accuracies, model_name, output_dir)
    logger.info(f"Cross-validation completed for {model_name}")


def main() -> tuple[float, float, float]:
    """Trains and evaluates models."""
    args = parse_args()
    logger.info(f"Running with arguments: {args}")

    # Determine which models to run
    models_to_run = (
        ["2D_CNN_aggr", "2D_CNN_late_fusion", "2D_CNN_early_fusion", "3D_CNN"] if args.model == "all" else [args.model]
    )

    final_train_acc = 0.0
    final_val_acc = 0.0
    final_test_acc = 0.0

    for model_name in models_to_run:
        # Override the model name in args to ensure proper model creation
        args.model = model_name

        if args.cross_validate:
            # Run cross-validation
            run_cross_validation(args, model_name)
        else:
            # Standard single train/test run
            logger.info(f"Training {model_name} for {args.num_epochs} epochs...")

            # Create dataloaders
            train_loader, val_loader, test_loader = create_dataloaders(args)

            # Create model
            model = create_model(args)

            # Train model
            train_acc, val_acc = model.fit(
                num_epochs=args.num_epochs, train_loader=train_loader, test_loader=val_loader
            )

            # Evaluate on test set
            test_acc = evaluate_on_videos(model, test_loader, model_name)

            save_results(model, train_acc, val_acc, test_acc, args)

            # Store the results for the return value
            final_train_acc = train_acc[-1]
            final_val_acc = val_acc[-1]
            final_test_acc = test_acc

    logger.info("Done!")

    # Return the results of the last model, or zeros if using cross-validation
    if args.cross_validate:
        return 0.0, 0.0, 0.0

    return final_train_acc, final_val_acc, final_test_acc


if __name__ == "__main__":
    main()

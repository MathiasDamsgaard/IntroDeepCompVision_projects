import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from datasets import FlowImageDataset, FlowVideoDataset, FrameImageDataset, FrameVideoDataset
from logger import logger
from model import CNN3D, BaselineClassifier, EarlyFusionCNN, FlowCNN, LateFusionCNN
from torch.utils.data import DataLoader
from torchvision import transforms

# Define a type for our models
ModelType = BaselineClassifier | LateFusionCNN | EarlyFusionCNN | CNN3D | FlowCNN


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
        choices=["2D_CNN_aggr", "2D_CNN_late_fusion", "2D_CNN_early_fusion", "3D_CNN", "Flow_CNN"],
        help="Model architecture to use",
    )

    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for optimizer")
    parser.add_argument("--image_size", type=int, default=64, help="Image size for resizing (square)")

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

    if args.model == "2D_CNN_aggr":
        # 2D models: frame-level training/val, video-level testing as list of frames
        frameimagetrain_dataset = FrameImageDataset(root_dir=root_dir, split="train", transform=transform)
        frameimageval_dataset = FrameImageDataset(root_dir=root_dir, split="val", transform=transform)
        framevideotest_dataset = FrameVideoDataset(
            root_dir=root_dir, split="test", transform=transform, stack_frames=False
        )

        train_loader = DataLoader(frameimagetrain_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(frameimageval_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(framevideotest_dataset, batch_size=args.batch_size, shuffle=False)

    elif args.model in {"2D_CNN_late_fusion", "2D_CNN_early_fusion"}:
        framevideotrain_dataset = FrameVideoDataset(
            root_dir=root_dir, split="train", transform=transform, stack_frames=False
        )
        framevideo_val_dataset = FrameVideoDataset(
            root_dir=root_dir, split="val", transform=transform, stack_frames=False
        )
        framevideo_test_dataset = FrameVideoDataset(
            root_dir=root_dir, split="test", transform=transform, stack_frames=False
        )

        train_loader = DataLoader(framevideotrain_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(framevideo_val_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(framevideo_test_dataset, batch_size=args.batch_size, shuffle=False)

    elif args.model == "3D_CNN":
        # Use video-level samples with frames stacked: [C, T, H, W]
        framevideotrain_dataset = FrameVideoDataset(
            root_dir=root_dir, split="train", transform=transform, stack_frames=True
        )
        framevideo_val_dataset = FrameVideoDataset(
            root_dir=root_dir, split="val", transform=transform, stack_frames=True
        )
        framevideo_test_dataset = FrameVideoDataset(
            root_dir=root_dir, split="test", transform=transform, stack_frames=True
        )

        train_loader = DataLoader(framevideotrain_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(framevideo_val_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(framevideo_test_dataset, batch_size=args.batch_size, shuffle=False)

    elif args.model == "Flow_CNN":
        flowimagetrain_dataset = FlowImageDataset(root_dir=root_dir, split="train", transform=transform)
        flowimageval_dataset = FlowImageDataset(root_dir=root_dir, split="val", transform=transform)
        flowvideotest_dataset = FlowVideoDataset(
            root_dir=root_dir, split="test", transform=transform, stack_frames=False
        )

        train_loader = DataLoader(flowimagetrain_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(flowimageval_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(flowvideotest_dataset, batch_size=args.batch_size, shuffle=True)

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
        model = CNN3D(n_classes=n_classes)
        model.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    elif args.model == "Flow_CNN":
        # Flow CNN model
        model = FlowCNN(n_classes=n_classes, lr=args.learning_rate, weight_decay=args.weight_decay)

    else:
        msg = f"Unknown model architecture: {args.model}"
        raise ValueError(msg)

    logger.info(f"Created {args.model} model")
    return model


def evaluate_on_videos(
    model: ModelType,
    test_loader: DataLoader,
) -> float:
    """Evaluate the model on complete test videos."""
    logger.info("Evaluating model on test videos...")
    n_correct = 0
    total_samples = 0

    for batch, targets in test_loader:
        batch_size = targets.size(0)
        total_samples += batch_size

        # Get model predictions for the batch
        predictions = model.evaluate(batch)

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
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    if args.no_leakage:
        plt.title(f"Model: {args.model} with no leakage - Test Accuracy: {test_acc:.2f}%")
    else:
        plt.title(f"Model: {args.model} with leakage - Test Accuracy: {test_acc:.2f}%")
    plt.legend()
    plt.grid(visible=True)
    plt.tight_layout()

    if args.no_leakage:
        plt_path = output_dir / f"{args.model}_accuracy_noleak.png"
    else:
        plt_path = output_dir / f"{args.model}_accuracy.png"
    plt.savefig(plt_path)
    logger.info(f"Accuracy plot saved to {plt_path}")


def main() -> tuple[float, float, float]:
    """Trains and evaluates models."""
    args = parse_args()
    logger.info(f"Running with arguments: {args}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(args)

    # Create model
    model = create_model(args)

    # Train model
    logger.info(f"Training {args.model} for {args.num_epochs} epochs...")
    train_acc, val_acc = model.fit(num_epochs=args.num_epochs, train_loader=train_loader, test_loader=val_loader)

    # Evaluate on test set
    logger.info(f"Evaluating {args.model} on test set...")
    test_acc = evaluate_on_videos(model, test_loader, args.model)

    # Save results
    logger.info("Saving results...")
    save_results(model, train_acc, val_acc, test_acc, args)

    logger.info("Done!")

    return train_acc, val_acc, test_acc


if __name__ == "__main__":
    main()

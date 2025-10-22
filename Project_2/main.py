import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from datasets import FrameImageDataset, FrameVideoDataset
from logger import logger
from model import CNN3D, BaselineClassifier, EarlyFusionCNN, LateFusionCNN
from torch.utils.data import DataLoader
from torchvision import transforms


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
        choices=["2D_CNN_aggr", "2D_CNN_late_fusion", "2D_CNN_early_fusion", "3D_CNN"],
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
    args.root_dir = str(root_path)

    if args.model == "3D_CNN":
        # Use video-level samples with frames stacked: [C, T, H, W]
        train_ds = FrameVideoDataset(root_dir=args.root_dir, split="train", transform=transform, stack_frames=True)
        val_ds = FrameVideoDataset(root_dir=args.root_dir, split="val", transform=transform, stack_frames=True)
        test_ds = FrameVideoDataset(root_dir=args.root_dir, split="test", transform=transform, stack_frames=True)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

    else:
        # 2D models: frame-level training/val, video-level testing as list of frames
        frameimagetrain_dataset = FrameImageDataset(root_dir=args.root_dir, split="train", transform=transform)
        frameimageval_dataset = FrameImageDataset(root_dir=args.root_dir, split="val", transform=transform)

        framevideotest_dataset = FrameVideoDataset(
            root_dir=args.root_dir, split="test", transform=transform, stack_frames=False
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
    model: BaselineClassifier | LateFusionCNN | EarlyFusionCNN | CNN3D,
    test_loader: DataLoader,
    args: argparse.Namespace,
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
        if args.model == "2D_CNN_aggr":
            # Process each frame independently and average predictions
            device = next(model.parameters()).device
            predictions = torch.zeros((batch_size, 10)).to(device)

            for i in range(n_frames):
                # Extract frame i from all videos in batch
                batch = frames_tensor[:, :, i, :, :].to(device)
                with torch.no_grad():
                    logits = model(batch)
                predictions += torch.softmax(logits, dim=1)

            predictions /= n_frames  # Average over number of frames

        elif args.model == "2D_CNN_late_fusion":
            # Late fusion model needs a list of frame tensors
            # Use list comprehension to create a list of frame tensors
            frame_list = [frames_tensor[:, :, i, :, :] for i in range(n_frames)]

            with torch.no_grad():
                logits = model(frame_list)
            predictions = torch.softmax(logits, dim=1)

        elif args.model == "2D_CNN_early_fusion":
            # Early fusion expects all frames stacked in channels
            # Reshape from [batch, C, n_frames, H, W] to [batch, C*n_frames, H, W]
            batch_size, c, n_frames, h, w = frames_tensor.shape
            stacked_frames = frames_tensor.reshape(batch_size, c * n_frames, h, w)
            # Make sure the stacked_frames are on the same device as the model
            device = next(model.parameters()).device
            stacked_frames = stacked_frames.to(device)
            with torch.no_grad():
                logits = model(stacked_frames)
            predictions = torch.softmax(logits, dim=1)

        elif args.model == "3D_CNN":
            # For 3D CNNs, pass the whole clip [B, C, T, H, W]
            device = next(model.parameters()).device
            with torch.no_grad():
                logits = model(frames_tensor.to(device))
            predictions = torch.softmax(logits, dim=1)

        else:
            msg = f"Evaluation not implemented for model: {args.model}"
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
    model: BaselineClassifier | LateFusionCNN | EarlyFusionCNN | CNN3D,
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
    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.axhline(y=test_acc / 100, color="r", linestyle="-", label=f"Test Accuracy: {test_acc:.2f}%")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    if args.no_leakage:
        plt.title(f"Model: {args.model} with no leakage")
    else:
        plt.title(f"Model: {args.model}")
    plt.legend()
    plt.grid(visible=True)

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

    # Train the model
    logger.info(f"Training {args.model} for {args.num_epochs} epochs...")

    train_acc, val_acc = model.fit(num_epochs=args.num_epochs, train_loader=train_loader, test_loader=val_loader)

    # Evaluate on test set
    test_acc = evaluate_on_videos(model, test_loader, args)

    # Save results
    save_results(model, train_acc, val_acc, test_acc, args)

    logger.info("Done!")

    return train_acc[-1], val_acc[-1], test_acc


if __name__ == "__main__":
    main()

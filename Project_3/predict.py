# Generate predictions on test set using a trained model
# Loads the saved weights and iterates over test set to generate segmentation masks

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset.drive_dataset import Drive
from dataset.ph2_dataset import Ph2
from model.encdec_model import EncDec
from model.unet_model import UNet
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Global seed for reproducibility (must match train.py)
RANDOM_SEED = 42


def load_model(
    model_name: str,
    checkpoint_path: str,
    device: torch.device,
) -> nn.Module:
    """Load trained model from checkpoint.

    Args:
        model_name: Name of model architecture ("EncDec" or "UNet")
        checkpoint_path: Path to model checkpoint file
        device: Device to load model on

    Returns:
        Loaded model in eval mode

    """
    models = {
        "EncDec": EncDec,
        "UNet": UNet,
    }

    if model_name not in models:
        msg = f"Unknown model: {model_name}"
        raise ValueError(msg)

    model = models[model_name]().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    return model


def create_data_loaders(
    dataset_name: str,
    size: int,
    batch_size: int,
    test_split: float,
    val_split: float,
    workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test data loaders by recreating the same splits as in training.

    Args:
        dataset_name: Name of dataset ("Drive" or "Ph2")
        size: Image size for resizing
        batch_size: Batch size for data loader
        test_split: Ratio for test set split (must match training)
        val_split: Ratio for validation set split (must match training)
        workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    """
    # Define transforms
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

    # Load dataset based on choice
    if dataset_name == "Drive":
        # No augmentation for prediction
        full_dataset = Drive(transform=transform)
    elif dataset_name == "Ph2":
        full_dataset = Ph2(transform=transform)
    else:
        msg = f"Unknown dataset: {dataset_name}"
        raise ValueError(msg)

    # Recreate the same splits as in training using the same seed
    total_size = len(full_dataset)
    test_size = int(test_split * total_size)
    train_val_size = total_size - test_size

    # First split: separate test set
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    all_indices_shuffled = torch.randperm(total_size, generator=generator).tolist()
    train_val_indices = all_indices_shuffled[:train_val_size]
    test_indices = all_indices_shuffled[train_val_size:]

    # Second split: separate validation from training
    val_size = int(val_split * train_val_size)
    train_size = train_val_size - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    shuffled = torch.randperm(train_val_size, generator=generator).tolist()
    train_indices = [train_val_indices[i] for i in shuffled[:train_size]]
    val_indices = [train_val_indices[i] for i in shuffled[train_size:]]

    # Create subsets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    return train_loader, val_loader, test_loader


def save_mask(array: np.ndarray, path: str) -> None:
    """Save binary mask as PNG image.

    Args:
        array: 2D numpy array with 0s and 1s
        path: Output file path

    """
    im_arr = array * 255
    Image.fromarray(np.uint8(im_arr)).save(path)


def visualize_predictions(
    images: list,
    ground_truths: list,
    predictions: list,
    output_dir: Path,
    indices_to_visualize: list[int] | None = None,
    num_samples: int = 3,
) -> None:
    """Create side-by-side visualizations of images, ground truth, and predictions.

    Args:
        images: List of input images (as tensors)
        ground_truths: List of ground truth masks (as numpy arrays)
        predictions: List of predicted masks (as numpy arrays)
        output_dir: Directory to save visualizations
        indices_to_visualize: Specific indices to visualize (if None, use first num_samples)
        num_samples: Number of samples to visualize if indices_to_visualize is None

    """
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # Determine which indices to visualize
    if indices_to_visualize is None:
        indices_to_visualize = list(range(min(num_samples, len(images))))

    for idx in indices_to_visualize:
        if idx >= len(images):
            continue

        img = images[idx]
        gt = ground_truths[idx]
        pred = predictions[idx]

        # Create figure
        plt.figure(figsize=(12, 4))

        # Plot input image
        plt.subplot(1, 3, 1)
        plt.title("Input Image", fontsize=12)
        if img.shape[0] == 1:
            plt.imshow(img[0].cpu().numpy(), cmap="gray")
        else:
            plt.imshow(img.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")

        # Plot ground truth
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth", fontsize=12)
        plt.imshow(gt, cmap="gray")
        plt.axis("off")

        # Plot prediction
        plt.subplot(1, 3, 3)
        plt.title("Prediction", fontsize=12)
        plt.imshow(pred, cmap="gray")
        plt.axis("off")

        plt.tight_layout()

        # Save visualization
        out_path = vis_dir / f"vis_{idx:04d}.png"
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()


def generate_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    threshold: float = 0.5,
    batch_size: int = 1,
    create_visualizations: bool = True,
    num_visualizations: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions on a dataset and save masks.

    Args:
        model: Trained model
        data_loader: DataLoader for the data (train/val/test)
        device: Device to run inference on
        output_dir: Directory to save prediction masks
        threshold: Threshold for binary segmentation
        batch_size: Batch size used in loader (for indexing)
        create_visualizations: Whether to create side-by-side visualizations
        num_visualizations: Number of samples to visualize

    Returns:
        Tuple of (predictions, ground_truths) as numpy arrays

    """
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(exist_ok=True)

    all_predictions = []
    all_ground_truths = []
    all_images = []

    with torch.no_grad():
        for i, (x, y_true) in enumerate(tqdm(data_loader, desc="Predicting")):
            x_gpu = x.to(device)

            # Get model output (logits)
            logits = model(x_gpu)

            # Convert to probabilities
            probs = torch.sigmoid(logits)

            # Apply threshold to get binary mask
            masks = (probs > threshold).cpu().numpy()

            # Save each mask in batch
            for j in range(masks.shape[0]):
                idx = i * batch_size + j
                mask = masks[j, 0]  # Shape: (H, W)
                save_mask(mask.astype(np.uint8), str(mask_dir / f"{idx:04d}.png"))

                # Store for metrics calculation and visualization
                all_predictions.append(mask)
                all_ground_truths.append(y_true[j, 0].cpu().numpy())
                all_images.append(x[j].cpu())

    predictions = np.array(all_predictions)
    ground_truths = np.array(all_ground_truths)

    # Save as numpy arrays for metric calculation
    np.save(output_dir / "predictions.npy", predictions)
    np.save(output_dir / "ground_truths.npy", ground_truths)

    # Create visualizations if requested
    if create_visualizations:
        print(f"Creating visualizations for {num_visualizations} samples...")  # noqa: T201
        visualize_predictions(
            images=all_images,
            ground_truths=all_ground_truths,
            predictions=all_predictions,
            output_dir=output_dir,
            num_samples=num_visualizations,
        )

    return predictions, ground_truths


def main() -> None:
    """Run prediction generation on train, val, and test sets."""
    parser = argparse.ArgumentParser(description="Generate predictions on train, val, and test sets")
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["Drive", "Ph2"], help="Dataset to use (Drive or Ph2)"
    )
    parser.add_argument(
        "--model", type=str, required=True, choices=["EncDec", "UNet"], help="Model architecture to use"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth file)")
    parser.add_argument("--size", type=int, default=128, help="Image size (must match training size)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument(
        "--output_dir", type=str, default="dataset/predictions", help="Directory to save prediction masks"
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary segmentation")
    parser.add_argument("--test_split", type=float, default=0.10, help="Test split ratio (must match training)")
    parser.add_argument("--val_split", type=float, default=0.10, help="Validation split ratio (must match training)")
    parser.add_argument("--workers", type=int, default=3, help="Number of data loading workers")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Which split to generate predictions for (default: all)",
    )
    parser.add_argument("--visualize", action="store_true", help="Create side-by-side visualizations of predictions")
    parser.add_argument("--num_vis", type=int, default=5, help="Number of samples to visualize (default: 5)")
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name=args.dataset,
        size=args.size,
        batch_size=args.batch_size,
        test_split=args.test_split,
        val_split=args.val_split,
        workers=args.workers,
    )

    # Determine which splits to process
    splits_to_process = []
    if args.split == "all":
        splits_to_process = [("train", train_loader), ("val", val_loader), ("test", test_loader)]
    elif args.split == "train":
        splits_to_process = [("train", train_loader)]
    elif args.split == "val":
        splits_to_process = [("val", val_loader)]
    elif args.split == "test":
        splits_to_process = [("test", test_loader)]

    # Generate and save predictions for each split
    for split_name, loader in splits_to_process:
        output_dir = Path(args.output_dir) / split_name

        print(f"\nGenerating predictions for {split_name} split...")  # noqa: T201
        _predictions, _ground_truths = generate_predictions(
            model=model,
            data_loader=loader,
            device=device,
            output_dir=output_dir,
            threshold=args.threshold,
            batch_size=args.batch_size,
            create_visualizations=args.visualize,
            num_visualizations=args.num_vis,
        )
        print(f"Saved {split_name} predictions to {output_dir}")  # noqa: T201


if __name__ == "__main__":
    main()

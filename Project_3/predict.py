# Generate predictions on test set using a trained model
# Loads the saved weights and iterates over test set to generate segmentation masks

import argparse
from pathlib import Path

import numpy as np
import torch
from dataset.driveDataset import Drive
from dataset.ph2Dataset import Ph2
from model.EncDecModel import EncDec
from model.UNetModel import UNet
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


def create_test_loader(
    dataset_name: str,
    size: int,
    batch_size: int,
    test_split: float,
    workers: int,
) -> DataLoader:
    """Create test data loader by recreating the same split as in training.

    Args:
        dataset_name: Name of dataset ("Drive" or "Ph2")
        size: Image size for resizing
        batch_size: Batch size for data loader
        test_split: Ratio for test set split (must match training)
        workers: Number of data loading workers

    Returns:
        DataLoader for test set

    """
    # Define transforms
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

    # Load dataset based on choice
    if dataset_name == "Drive":
        # Test set has no labels - use train for all splits (same as training)
        full_dataset = Drive(train=True, transform=transform)
    elif dataset_name == "Ph2":
        full_dataset = Ph2(transform=transform)
    else:
        msg = f"Unknown dataset: {dataset_name}"
        raise ValueError(msg)

    # Recreate the same split as in training using the same seed
    total_size = len(full_dataset)
    test_size = int(test_split * total_size)
    train_val_size = total_size - test_size

    # Split with same seed as training to get exact same test set
    _, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_val_size, test_size], generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    # Create data loader
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)


def save_mask(array: np.ndarray, path: str) -> None:
    """Save binary mask as PNG image.

    Args:
        array: 2D numpy array with 0s and 1s
        path: Output file path

    """
    im_arr = array * 255
    Image.fromarray(np.uint8(im_arr)).save(path)


def generate_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    threshold: float = 0.5,
    batch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions on test set and save masks.

    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run inference on
        output_dir: Directory to save prediction masks
        threshold: Threshold for binary segmentation
        batch_size: Batch size used in loader (for indexing)

    Returns:
        Tuple of (predictions, ground_truths) as numpy arrays

    """
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(exist_ok=True)

    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for i, (x, y_true) in enumerate(tqdm(test_loader, desc="Predicting")):
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

                # Store for metrics calculation
                all_predictions.append(mask)
                all_ground_truths.append(y_true[j, 0].cpu().numpy())

    predictions = np.array(all_predictions)
    ground_truths = np.array(all_ground_truths)

    # Save as numpy arrays for metric calculation
    np.save(output_dir / "predictions.npy", predictions)
    np.save(output_dir / "ground_truths.npy", ground_truths)

    return predictions, ground_truths


def main() -> None:
    """Run prediction generation on test set."""
    parser = argparse.ArgumentParser(description="Generate predictions on test set")
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
    parser.add_argument("--workers", type=int, default=3, help="Number of data loading workers")
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # Create test data loader
    test_loader = create_test_loader(
        dataset_name=args.dataset,
        size=args.size,
        batch_size=args.batch_size,
        test_split=args.test_split,
        workers=args.workers,
    )

    # Generate and save predictions
    output_dir = Path(args.output_dir)
    _predictions, _ground_truths = generate_predictions(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

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
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm


def save_mask(array: np.ndarray, path: str) -> None:
    """Save binary mask as PNG image."""
    # array should be a 2D numpy array with 0s and 1s
    im_arr = array * 255
    Image.fromarray(np.uint8(im_arr)).save(path)


def main() -> None:
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
    parser.add_argument("--output_dir", type=str, default="predictions", help="Directory to save prediction masks")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary segmentation")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    models = {
        "EncDec": EncDec,
        "UNet": UNet,
    }
    model = models[args.model]().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Load test dataset
    transform = transforms.Compose([transforms.Resize((args.size, args.size)), transforms.ToTensor()])

    if args.dataset == "Drive":
        # Drive has a separate test set (without labels)
        # We'll use the training set for evaluation since test has no labels
        testset = Drive(train=True, transform=transform)
    elif args.dataset == "Ph2":
        # Ph2: Load test split using saved indices
        test_indices_path = Path("splits/ph2_test_indices.pt")
        if not test_indices_path.exists():
            msg = f"Test indices not found at {test_indices_path}. Please run training first to create the splits."
            raise FileNotFoundError(msg)

        full_dataset = Ph2(transform=transform)
        test_indices = torch.load(test_indices_path)
        testset = Subset(full_dataset, test_indices)

    loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(exist_ok=True)

    # Generate predictions
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for i, (x, y_true) in enumerate(tqdm(loader)):
            x_gpu = x.to(device)

            # Get model output (logits)
            logits = model(x_gpu)

            # Convert to probabilities
            probs = torch.sigmoid(logits)

            # Apply threshold to get binary mask
            masks = (probs > args.threshold).cpu().numpy()

            # Save each mask in batch
            for j in range(masks.shape[0]):
                idx = i * args.batch_size + j
                mask = masks[j, 0]  # Shape: (H, W)
                save_mask(mask.astype(np.uint8), str(mask_dir / f"{idx:04d}.png"))

                # Store for metrics calculation
                all_predictions.append(mask)
                all_ground_truths.append(y_true[j, 0].cpu().numpy())

    # Save predictions and ground truths as numpy arrays for metric calculation
    np.save(output_dir / "predictions.npy", np.array(all_predictions))
    np.save(output_dir / "ground_truths.npy", np.array(all_ground_truths))


if __name__ == "__main__":
    main()

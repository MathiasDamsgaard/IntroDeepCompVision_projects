import argparse

import torch
from datasets import FrameImageDataset, FrameVideoDataset
from logger import logger
from model import BaselineClassifier
from torch.utils.data import DataLoader
from torchvision import transforms

logger.info("Starting training...")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a video classifier on UFC10 dataset")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    args = parser.parse_args()

    logger.info(f"Training configuration: {args}")

    # Fixed parameters
    n_classes = 10
    n_frames = 10
    batch_size = 8
    image_size = (64, 64)

    root_dir = "/dtu/datasets1/02516/ufc10"

    logger.info(f"Loading data from {root_dir}...")

    # Resize images to 64x64 and convert to tensors
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

    # FrameImageDataset: Returns individual frames for training/validation
    frameimagetrain_dataset = FrameImageDataset(root_dir=root_dir, split="train", transform=transform)
    frameimageval_dataset = FrameImageDataset(root_dir=root_dir, split="val", transform=transform)

    # FrameVideoDataset: Returns all frames of a video as a list (stack_frames=False)
    framevideotest_dataset = FrameVideoDataset(root_dir=root_dir, split="test", transform=transform, stack_frames=False)

    frameimagetrain_loader = DataLoader(frameimagetrain_dataset, batch_size=batch_size, shuffle=True)
    frameimageval_loader = DataLoader(frameimageval_dataset, batch_size=batch_size, shuffle=True)
    framevideotest_loader = DataLoader(framevideotest_dataset, batch_size=batch_size, shuffle=True)

    model = BaselineClassifier(n_classes=n_classes)

    # Train on individual frames from the training set, validate on individual frames
    train_acc, test_acc = model.fit(
        num_epochs=args.num_epochs, train_loader=frameimagetrain_loader, test_loader=frameimageval_loader
    )
    # Evaluate on complete videos by averaging predictions across all frames
    n_correct = 0
    for frames, targets in framevideotest_loader:
        batch_size = targets.size(0)
        # Initialize prediction accumulator for this batch (reset for each batch!)
        predictions = torch.zeros((batch_size, n_frames))

        # Average predictions across all frames in each video
        # frames is a list of 10 frame batches, each with shape [batch_size, C, H, W]
        for batch in frames:
            logits = model.evaluate(batch)  # Get raw logits from model
            # Convert logits to probabilities before averaging (important!)
            predictions += torch.softmax(logits, dim=1)

        predictions /= len(frames)  # Average over number of frames (10 frames per video)

        # Final prediction is the class with highest average probability
        n_correct += (predictions.argmax(1) == targets).sum().item()

    logger.info(f"Test accuracy on complete videos: {n_correct / len(framevideotest_dataset):.4f}")
    logger.info("Training complete.")

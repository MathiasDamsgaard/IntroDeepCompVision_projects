import argparse

import torch
from datasets import FlowImageDataset, FlowVideoDataset, FrameImageDataset, FrameVideoDataset
from logger import logger
from model import FlowCNN
from torch.utils.data import DataLoader
from torchvision import transforms

logger.info("Starting training...")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a video classifier on UFC10 dataset")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs (default: 1)")
    args = parser.parse_args()

    logger.info(f"Training configuration: {args}")

    # Fixed parameters
    n_classes = 10
    n_frames = 10
    batch_size = 8
    image_size = (64, 64)

    root_dir = "/dtu/datasets1/02516/ufc10"
    noleakage_dir = "/dtu/datasets1/02516/ucf101_noleakage"

    logger.info(f"Loading data from {root_dir}...")

    # Resize images to 64x64 and convert to tensors
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

    # FrameImageDataset: Returns individual frames for training/validation
    frameimagetrain_dataset = FrameImageDataset(root_dir=root_dir, split="train", transform=transform)
    frameimageval_dataset = FrameImageDataset(root_dir=root_dir, split="val", transform=transform)
    frameimagetest_dataset = FrameImageDataset(root_dir=root_dir, split="test", transform=transform)

    # FrameVideoDataset: Returns all frames of a video as a list (stack_frames=False)
    framevideotrain_dataset = FrameVideoDataset(
        root_dir=root_dir, split="train", transform=transform, stack_frames=False
    )
    framevideoval_dataset = FrameVideoDataset(root_dir=root_dir, split="val", transform=transform, stack_frames=False)
    framevideotest_dataset = FrameVideoDataset(root_dir=root_dir, split="test", transform=transform, stack_frames=False)

    # FrameVideoDataset: For 3D CNN we want stacked frames [C, T, H, W]
    framevideotrainstack_dataset = FrameVideoDataset(
        root_dir=root_dir, split="train", transform=transform, stack_frames=True
    )
    framevideovalstack_dataset = FrameVideoDataset(
        root_dir=root_dir, split="val", transform=transform, stack_frames=True
    )
    framevideoteststack_dataset = FrameVideoDataset(
        root_dir=root_dir, split="test", transform=transform, stack_frames=True
    )

    # DataLoaders
    frameimagetrain_loader = DataLoader(frameimagetrain_dataset, batch_size=batch_size, shuffle=True)
    frameimageval_loader = DataLoader(frameimageval_dataset, batch_size=batch_size, shuffle=True)
    frameimagetest_loader = DataLoader(frameimagetest_dataset, batch_size=batch_size, shuffle=True)

    framevideotrain_loader = DataLoader(framevideotrain_dataset, batch_size=batch_size, shuffle=True)
    framevideoval_loader = DataLoader(framevideoval_dataset, batch_size=batch_size, shuffle=True)
    framevideotest_loader = DataLoader(framevideotest_dataset, batch_size=batch_size, shuffle=True)

    framevideotrainstack_loader = DataLoader(framevideotrainstack_dataset, batch_size=batch_size, shuffle=True)
    framevideovalstack_loader = DataLoader(framevideovalstack_dataset, batch_size=batch_size, shuffle=True)
    framevideoteststack_loader = DataLoader(framevideoteststack_dataset, batch_size=batch_size, shuffle=True)

    # Flow datasets
    flowimagetrain_dataset = FlowImageDataset(root_dir=noleakage_dir, split="train", transform=transform)
    flowimageval_dataset = FlowImageDataset(root_dir=noleakage_dir, split="val", transform=transform)
    flowvideotest_dataset = FlowVideoDataset(
        root_dir=noleakage_dir, split="test", transform=transform, stack_frames=False
    )

    train_loader = DataLoader(flowimagetrain_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(flowimageval_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(flowvideotest_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    logger.info("Creating model...")
    model = FlowCNN(n_classes=n_classes, n_channels=3)

    # Train on individual frames from the training set, validate on individual frames
    logger.info(f"Starting training for {args.num_epochs} epochs...")
    train_acc, test_acc = model.fit(num_epochs=args.num_epochs, train_loader=train_loader, test_loader=val_loader)

    logger.info("Evaluating on test set...")
    for batch, _target in test_loader:
        predictions = model.evaluate(batch)

    # Evaluate on complete videos by averaging predictions across all frames
    n_correct = 0
    # for frames, targets in framevideotest_loader:
    #     batch_size = targets.size(0)
    #     # Initialize prediction accumulator for this batch (reset for each batch!)
    #     predictions = torch.zeros((batch_size, n_frames))

    #     # Average predictions across all frames in each video
    #     # frames is a list of 10 frame batches, each with shape [batch_size, C, H, W]
    #     for batch in frames:
    #         logits = model.evaluate(batch)  # Get raw logits from model
    #         # Convert logits to probabilities before averaging (important!)
    #         predictions += torch.softmax(logits, dim=1)

    #     predictions /= len(frames)  # Average over number of frames (10 frames per video)

    #     # Final prediction is the class with highest average probability
    #     n_correct += (predictions.argmax(1) == targets).sum().item()

    # logger.info(f"Test accuracy on complete videos: {n_correct / len(framevideotest_dataset):.4f}")

    # Evaluate on complete videos directly as clips [B, C, T, H, W]
    device = next(model.parameters()).device
    n_total = 0
    for clips, targets in framevideoteststack_loader:
        with torch.no_grad():
            logits = model(clips.to(device))
            preds = logits.argmax(1)
        n_correct += (preds.cpu() == targets).sum().item()
        n_total += targets.size(0)
    logger.info(f"Test accuracy on complete videos: {n_correct / n_total:.4f}")

    logger.info("Training complete.")

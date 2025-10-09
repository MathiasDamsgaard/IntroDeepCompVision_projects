from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch
import argparse

from datasets import FrameImageDataset, FrameVideoDataset
from model import VideoClassifier

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a video classifier on UFC10 dataset')
    parser.add_argument('--num_epochs', type=int, default=5, 
                        help='Number of training epochs (default: 5)')
    args = parser.parse_args()

    # Fixed parameters
    n_classes = 10
    n_frames = 10
    batch_size = 8
    image_size = (64, 64)

    root_dir = '/dtu/datasets1/02516/ufc10'

    # Resize images to 64x64 and convert to tensors
    transform = T.Compose([T.Resize(image_size), T.ToTensor()])

    print("Loading datasets...")
    # FrameImageDataset: Returns individual frames for training/validation
    frameimagetrain_dataset = FrameImageDataset(root_dir=root_dir, split='train', transform=transform)
    frameimageval_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    
    # FrameVideoDataset: Returns all frames of a video as a list (stack_frames=False)
    framevideotest_dataset = FrameVideoDataset(root_dir=root_dir, split='test',
                                               transform=transform, stack_frames = False)
    
    print("Loading dataloaders...")
    frameimagetrain_loader = DataLoader(frameimagetrain_dataset,  batch_size=batch_size, shuffle=True)
    frameimageval_loader = DataLoader(frameimageval_dataset,  batch_size=batch_size, shuffle=True)
    framevideotest_loader = DataLoader(framevideotest_dataset,  batch_size=batch_size, shuffle=True)

    print("Loading model...")
    model = VideoClassifier(n_classes=n_classes)

    print(f"Training model for {args.num_epochs} epochs...")
    # Train on individual frames from the training set, validate on individual frames
    train_acc, test_acc = model.train(num_epochs=args.num_epochs, train_loader=frameimagetrain_loader,
                                      test_loader=frameimageval_loader)
    
    print("Evaluating model...")
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

    print(f"Test accuracy: {n_correct / len(framevideotest_loader.dataset) * 100:.2f}%")
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch

from datasets import FrameImageDataset, FrameVideoDataset
from model import VideoClassifier

if __name__ == '__main__':
    root_dir = '/dtu/datasets1/02516/ufc10'

    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])

    print("Loading datasets...")
    frameimagetrain_dataset = FrameImageDataset(root_dir=root_dir, split='train', transform=transform)
    frameimageval_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    framevideotest_dataset = FrameVideoDataset(root_dir=root_dir, split='test',
                                               transform=transform, stack_frames = False)
    print("Loading dataloaders...")
    frameimagetrain_loader = DataLoader(frameimagetrain_dataset,  batch_size=8, shuffle=True)
    frameimageval_loader = DataLoader(frameimageval_dataset,  batch_size=8, shuffle=True)
    framevideotest_loader = DataLoader(framevideotest_dataset,  batch_size=8, shuffle=True)
    
    print("Loading model...")
    model = VideoClassifier(n_classes=10)

    print("Training model...")
    train_acc, test_acc = model.train(num_epochs=5, train_loader=frameimagetrain_loader,
                                      test_loader=frameimageval_loader)
    
    print("Evaluating model...")
    n_correct = 0
    predictions = torch.zeros((8, 10))
    for frames, targets in framevideotest_loader:
        for batch in frames:
            predictions += model.evaluate(batch)

        n_correct += (predictions.argmax(1) == targets).sum().item()

    print(f"Test accuracy: {n_correct / len(framevideotest_loader.dataset) * 100:.2f}%")
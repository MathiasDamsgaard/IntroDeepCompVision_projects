import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


class ProposalDataset(Dataset):
    """Dataset for object proposals."""

    def __init__(
        self,
        pickle_path: str,
        image_dir: str,
        transform: transforms.Compose | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            pickle_path (str): Path to the pickle file containing proposals.
            image_dir (str): Directory containing images.
            transform (transforms.Compose | None, optional): Transforms to apply. Defaults to None.

        """
        self.image_dir = Path(image_dir)
        self.transform = transform

        with Path(pickle_path).open("rb") as f:
            self.data = pickle.load(f)  # noqa: S301

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.data[idx]
        img_name = item["image"]
        x, y, w, h = item["proposal"]
        label = item["label"]

        img_path = self.image_dir / img_name
        # Use PIL to be consistent with torchvision transforms
        img = Image.open(img_path).convert("RGB")

        # Crop the proposal
        # PIL crop: (left, upper, right, lower)
        img_crop = img.crop((x, y, x + w, y + h))
        img_crop = self.transform(img_crop) if self.transform else transforms.ToTensor()(img_crop)

        # Ensure img_crop is a Tensor for type safety
        if not isinstance(img_crop, torch.Tensor):
            img_crop = transforms.ToTensor()(img_crop)

        return img_crop, int(label)


def get_dataloader(
    pickle_path: str,
    image_dir: str,
    batch_size: int = 32,
    is_train: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader for the dataset.

    Args:
        pickle_path (str): Path to the pickle file.
        image_dir (str): Directory containing images.
        batch_size (int, optional): Batch size. Defaults to 32.
        is_train (bool, optional): Whether to use weighted sampling for training. Defaults to True.
        num_workers (int, optional): Number of workers. Defaults to 4.

    Returns:
        DataLoader: The DataLoader.

    """
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ProposalDataset(pickle_path, image_dir, transform=transform)

    if is_train:
        # Calculate weights for WeightedRandomSampler
        targets = [item["label"] for item in dataset.data]
        class_counts = np.bincount(targets)

        # We want 25% positive (class 1) and 75% negative (class 0)
        # Weight for class i: W_i = P_target(i) / N_i
        # W_0 = 0.75 / N_0
        # W_1 = 0.25 / N_1

        weights = []
        # Handle case where a class might be missing (though unlikely for train set)
        w0 = 0.75 / class_counts[0] if class_counts[0] > 0 else 0.0
        w1 = 0.25 / class_counts[1] if len(class_counts) > 1 and class_counts[1] > 0 else 0.0

        for t in targets:
            if t == 0:
                weights.append(w0)
            else:
                weights.append(w1)

        # Set samples to a reasonable number to avoid excessive iterations per epoch.
        # 10000 gives ~312 batches of size 32.
        num_samples = 10000
        # num_samples = len(weights)
        sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)

        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

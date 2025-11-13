from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

DATA_PATH = "/dtu/datasets1/02516/DRIVE"


class Drive(torch.utils.data.Dataset):
    def __init__(self, transform: transforms.Compose) -> None:
        """Initialize the dataset.

        Args:
            transform: Standard transform to apply

        """
        self.transform = transform
        data_path = Path.joinpath(Path(DATA_PATH), "training")
        self.image_paths = sorted(Path.glob(data_path / "images", "*.tif"))
        self.label_paths = sorted(Path.glob(data_path / "1st_manual", "*.gif"))

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[Image.Image, Image.Image]:
        """Generate one sample of data."""
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        label_path = self.label_paths[idx]
        label = Image.open(label_path)

        x = self.transform(image)
        y = self.transform(label)

        return x, y

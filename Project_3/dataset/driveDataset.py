from pathlib import Path  # noqa: N999

import torch
from PIL import Image
from torchvision import transforms

DATA_PATH = "/dtu/datasets1/02516/DRIVE"


class Drive(torch.utils.data.Dataset):
    def __init__(self, train: bool, transform: transforms.Compose) -> None:
        """Initialize the dataset."""
        self.transform = transform
        data_path = Path.joinpath(Path(DATA_PATH), "train" if train else "test")
        self.image_paths = sorted(Path.glob(data_path / "images", "*.tif"))
        self.mask_paths = sorted(Path.glob(data_path / "mask", "*.gif"))

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[Image.Image, Image.Image]:
        """Generate one sample of data."""
        image_path = self.image_paths[idx]
        label_path = self.mask_paths[idx]

        image = Image.open(image_path)
        label = Image.open(label_path)
        y = self.transform(label)
        x = self.transform(image)
        return x, y

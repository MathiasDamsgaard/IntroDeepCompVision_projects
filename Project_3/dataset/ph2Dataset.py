# noqa: N999
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

DATA_PATH = "/dtu/datasets1/02516/PH2_Dataset_images"


class Ph2(torch.utils.data.Dataset):
    def __init__(self, transform: transforms.Compose) -> None:
        """Initialize the dataset."""
        self.transform = transform
        data_path = Path(DATA_PATH)
        all_paths = sorted(Path.glob(data_path, "/*"))
        self.image_paths = []
        self.lesion_paths = []
        for path in all_paths:
            paths = sorted(Path.glob(path, "/*/*.bmp"))
            for p in paths:
                if "Dermoscopic_Image" in str(p):
                    self.image_paths.append(p)
                elif "lesion" in str(p):
                    self.lesion_paths.append(p)

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[Image.Image, Image.Image]:
        """Generate one sample of data."""
        image_path = self.image_paths[idx]
        lesion_path = self.lesion_paths[idx]

        image = Image.open(image_path)
        lesion = Image.open(lesion_path)
        y = self.transform(lesion)
        x = self.transform(image)
        return x, y

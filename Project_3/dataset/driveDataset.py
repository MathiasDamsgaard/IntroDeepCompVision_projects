import glob
import os

import torch
from PIL import Image

DATA_PATH = "/dtu/datasets1/02516/DRIVE"


class Drive(torch.utils.data.Dataset):
    def __init__(self, train, transform) -> None:
        """Initialization."""
        self.transform = transform
        data_path = os.path.join(DATA_PATH, "train" if train else "test")
        self.image_paths = sorted(glob.glob(data_path + "/images/*.tif"))
        self.mask_paths = sorted(glob.glob(data_path + "/mask/*.gif"))

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Generates one sample of data."""
        image_path = self.image_paths[idx]
        label_path = self.mask_paths[idx]

        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = self.transform(label)
        X = self.transform(image)
        return X, Y

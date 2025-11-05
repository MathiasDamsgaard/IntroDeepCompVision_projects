import glob

import torch
from PIL import Image

DATA_PATH = "/dtu/datasets1/02516/PH2_Dataset_images"


class Ph2(torch.utils.data.Dataset):
    def __init__(self, transform) -> None:
        """Initialization."""
        self.transform = transform
        data_path = DATA_PATH
        all_paths = sorted(glob.glob(data_path + "/*"))
        self.image_paths = []
        self.lesion_paths = []
        for path in all_paths:
            paths = glob.glob(path + "/*/*.bmp")
            for p in paths:
                if "Dermoscopic_Image" in p:
                    self.image_paths.append(p)
                elif "lesion" in p:
                    self.lesion_paths.append(p)

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Generates one sample of data."""
        image_path = self.image_paths[idx]
        lesion_path = self.lesion_paths[idx]

        image = Image.open(image_path)
        lesion = Image.open(lesion_path)
        Y = self.transform(lesion)
        X = self.transform(image)
        return X, Y

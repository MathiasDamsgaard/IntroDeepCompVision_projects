# noqa: N999
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from utils import generate_clicks 

DATA_PATH = "/dtu/datasets1/02516/PH2_Dataset_images"


class Ph2(torch.utils.data.Dataset):
    def __init__(self, transform: transforms.Compose) -> None:
        """Initialize the dataset."""
        self.transform = transform
        data_path = Path(DATA_PATH)
        # Use glob on the Path object instead of Path.glob() for Python 3.13 compatibility
        all_paths = sorted(data_path.glob("*"))
        self.image_paths = []
        self.lesion_paths = []
        for path in all_paths:
            # Use glob on the path object directly
            paths = sorted(path.glob("*/*.bmp"))
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





class WeaklySupervisedPh2(Ph2):
    """
    A version of the Ph2 dataset that returns simulated clicks instead of full masks.
    """
    def __init__(self, transform, num_pos_clicks=10, num_neg_clicks=10):
        # Initialize the parent Ph2 class
        super().__init__(transform=transform)
        self.num_pos_clicks = num_pos_clicks
        self.num_neg_clicks = num_neg_clicks

    def __getitem__(self, idx):
        # Get the original image and the full mask from the parent class
        image, full_mask = super().__getitem__(idx)
        
        # Now, generate the click mask from the full mask
        click_mask = generate_clicks(
            full_mask, 
            num_pos_clicks=self.num_pos_clicks, 
            num_neg_clicks=self.num_neg_clicks
        )
        
        # Return the image and the click_mask (instead of the full_mask)
        return image, click_mask
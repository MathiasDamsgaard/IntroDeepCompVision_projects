import numpy as np
import torch


def generate_clicks(mask: torch.Tensor, num_pos_clicks: int = 5, num_neg_clicks: int = 5) -> torch.Tensor:
    """Run simulations of user clicks by sampling points from a full segmentation mask.

    Args:
        mask (torch.Tensor): The ground truth mask, shape [1, H, W].
        num_pos_clicks (int): Number of positive clicks to generate.
        num_neg_clicks (int): Number of negative clicks to generate.

    Returns:
        torch.Tensor: A "click mask" of the same shape as the input, with:
                      - 1 at positive click locations
                      - 0 at negative click locations
                      - -100 (an ignore value) everywhere else

    """
    # Create an empty click mask with the ignore value
    click_mask = torch.full_like(mask, -100)

    # Find coordinates of positive (lesion) and negative (background) pixels
    pos_coords = (mask == 1).nonzero()
    neg_coords = (mask == 0).nonzero()

    # --- Sample Positive Clicks ---
    if len(pos_coords) > 0:
        # Randomly choose indices for the positive clicks
        pos_indices = np.random.choice(len(pos_coords), size=num_pos_clicks, replace=True)  # noqa: NPY002
        for idx in pos_indices:
            # Get the coordinate and set the value in the click mask to 1
            coord = pos_coords[idx]
            click_mask[coord[0], coord[1], coord[2]] = 1

    # --- Sample Negative Clicks ---
    if len(neg_coords) > 0:
        neg_indices = np.random.choice(len(neg_coords), size=num_neg_clicks, replace=True)  # noqa: NPY002
        for idx in neg_indices:
            coord = neg_coords[idx]
            click_mask[coord[0], coord[1], coord[2]] = 0

    return click_mask

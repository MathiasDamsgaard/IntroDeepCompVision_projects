from pathlib import Path

import matplotlib as mpl
import numpy as np
from PIL import Image
from torchvision import transforms

mpl.use("Agg")  # use non-GUI backend
import matplotlib.pyplot as plt
from dataset.PhCDataset import PhC

# --- config ---
size = 128
indices_to_visualize = [1, 2, 3]  # pick whatever indices you want
pred_dir = Path("preds")  # folder with your preds
out_dir = Path("vis")  # folder to save visualizations
out_dir.mkdir(exist_ok=True)

# --- dataset (must match train/test setup) ---
transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

testset = PhC(train=False, transform=transform)

for idx in indices_to_visualize:
    if idx >= len(testset):
        continue

    img, y_true = testset[idx]
    # y_true: (1, H, W) in [0,1]; binarize and squeeze
    threshold = 0.5
    y_true = (y_true > threshold).float()[0].numpy()  # HxW

    pred_path = pred_dir / f"{idx:04d}.png"
    if not pred_path.exists():
        continue

    pred_img = Image.open(pred_path).convert("L").resize((size, size), resample=Image.NEAREST)
    pred_arr = np.array(pred_img)
    pred_bin = (pred_arr > threshold * 255).astype(np.float32)  # HxW

    # --- plot and save ---
    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.title("Image")
    # img: (C, H, W). If C=1, make it HxW for imshow
    if img.shape[0] == 1:
        plt.imshow(img[0].numpy(), cmap="gray")
    else:
        plt.imshow(img.permute(1, 2, 0).numpy(), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("GT mask")
    plt.imshow(y_true, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Pred mask")
    plt.imshow(pred_bin, cmap="gray")
    plt.axis("off")

    plt.tight_layout()

    out_path = out_dir / f"vis_{idx:04d}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

import argparse
from pathlib import Path
from time import time

import torch
from dataset.driveDataset import Drive
from dataset.ph2Dataset import Ph2
from losses import CrossEntropyLoss, FocalLoss, WeightedCrossEntropyLoss
from model.EncDecModel import EncDec
from model.UNetModel import UNet
from torch import optim
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train segmentation model")
parser.add_argument(
    "--dataset", type=str, default="Drive", choices=["Drive", "Ph2"], help="Dataset to use for training"
)
parser.add_argument("--model", type=str, default="EncDec", choices=["EncDec", "UNet"], help="Model architecture to use")
parser.add_argument(
    "--loss",
    type=str,
    default="CrossEntropyLoss",
    choices=["CrossEntropyLoss", "FocalLoss", "WeightedCrossEntropyLoss"],
    help="Loss function to use",
)
parser.add_argument("--batch_size", type=int, default=6, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
parser.add_argument("--size", type=int, default=128, help="Image size (images will be resized to size x size)")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--workers", type=int, default=3, help="Number of data loading workers")
parser.add_argument(
    "--pos_weight",
    type=float,
    default=None,
    help="Positive class weight for WeightedCrossEntropyLoss (auto-calculated if not provided)",
)
parser.add_argument(
    "--test_split", type=float, default=0.15, help="Test split ratio for Ph2 dataset (ignored for Drive)"
)
parser.add_argument(
    "--val_split", type=float, default=0.15, help="Validation split ratio (applied after test split for Ph2)"
)
args = parser.parse_args()

# Dataset
size = args.size
train_transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

batch_size = args.batch_size
workers = args.workers

# Load dataset based on choice
if args.dataset == "Drive":
    # Drive has predefined train/test splits
    # Only test set has no labels
    trainset = Drive(train=True, transform=train_transform)

    # Validation setup - split from training set
    val_size = int(args.val_split * len(trainset))
    train_size = len(trainset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(trainset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=workers)


elif args.dataset == "Ph2":
    # Ph2 has no predefined splits - we need to create train/val/test splits
    full_dataset = Ph2(transform=train_transform)
    total_size = len(full_dataset)

    # First split: separate test set
    test_size = int(args.test_split * total_size)
    train_val_size = total_size - test_size
    train_val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_val_size, test_size],
        generator=torch.Generator().manual_seed(42),  # Fixed seed for reproducibility
    )

    # Second split: separate validation from training
    val_size = int(args.val_split * train_val_size)
    train_size = train_val_size - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_val_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=workers)

    # Save test indices for later use in prediction
    splits_dir = Path("splits")
    splits_dir.mkdir(exist_ok=True)
    test_indices = test_dataset.indices
    torch.save(test_indices, splits_dir / "ph2_test_indices.pt")


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {
    "EncDec": EncDec,
    "UNet": UNet,
}
model = models[args.model]().to(device)

summary(model, (3, size, size))
learning_rate = args.lr
optimizer = optim.Adam(model.parameters(), learning_rate)

loss_fns = {
    "CrossEntropyLoss": CrossEntropyLoss,
    "FocalLoss": FocalLoss,
    "WeightedCrossEntropyLoss": WeightedCrossEntropyLoss,
}

# Instantiate loss function with appropriate parameters
if args.loss == "WeightedCrossEntropyLoss":
    loss_fn = loss_fns[args.loss](pos_weight=args.pos_weight)
else:
    loss_fn = loss_fns[args.loss]()

epochs = args.epochs


# Training loop
for _epoch in tqdm(range(epochs)):
    tic = time()

    # Training phase
    model.train()
    train_loss = 0
    for X_batch, y_true in train_loader:
        X_batch_gpu = X_batch.to(device)
        y_true_gpu = y_true.to(device)

        # set parameter gradients to zero
        optimizer.zero_grad()

        # forward + backward + optimize
        y_pred = model(X_batch_gpu)
        # IMPORTANT NOTE: Check whether y_pred is normalized or unnormalized
        # and whether it makes sense to apply sigmoid or softmax.
        loss = loss_fn(y_pred, y_true_gpu)  # forward-pass
        loss.backward()  # backward-pass
        optimizer.step()  # update weights

        # calculate metrics to show the user
        train_loss += loss.item() / len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_true in val_loader:
            X_batch_gpu = X_batch.to(device)
            y_true_gpu = y_true.to(device)

            y_pred = model(X_batch_gpu)
            loss = loss_fn(y_pred, y_true_gpu)
            val_loss += loss.item() / len(val_loader)

    toc = time()


# Save the model
checkpoint_dir = Path("model/checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
model_filename = checkpoint_dir / f"{args.dataset.lower()}_{args.model.lower()}_{args.loss.lower()}.pth"
torch.save(model.state_dict(), model_filename)
